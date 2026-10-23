"""Train the class labeller used to count generated images, with a background class.

The earlier probes were trained on dataset-0-v2's reference split: 225 images
per low class, reweighted 21x. On ambiguous generated images that stretched
the rare classes' regions (empty blue frames became marine_animal). This
labeller instead uses:

* ``--per-class`` real pool images per class (equal counts, no reweighting),
  excluding the unseen-real evaluation images of evaluate_v2.py;
* a ``background`` class of images with no object: corner patches of pool
  images upscaled to 64x64, and pool images under a strong Gaussian blur;
* concatenated, L2-normalised DINOv2 and CLIP ViT-L/14 features and a
  multinomial logistic regression.

It reports accuracy on the unseen real images and on held-out background
images, then labels the generated set and prints the bias metrics of
compare_labellers.py. The fitted model is saved for G1 and later runs.

Example::

    python train_labeller.py --data-dir ../data/dataset-0-v2 --pool-dir ../data/pool \\
        --generated G0v2=../results/samples_g0_v2:*.png --dino-cache ../results/eval_g0_v2/cache \\
        --clip-cache ../results/labellers_g0_v2/cache --output-dir ../results/labeller_v2
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchvision import utils

from compare_labellers import ClipFeatures, metrics
from evaluate_features import BACKBONE, Features
from evaluate_v2 import CLASSES, POOL_NAMES, split, training_images, unseen_pool

BACKGROUND = "background"
LABELS = CLASSES + [BACKGROUND]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--generated", required=True, metavar="NAME=DIR:GLOB")
    parser.add_argument("--dino-cache", type=Path, required=True)
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=3000)
    parser.add_argument("--background", type=int, default=3600, help="Background images (train + held out)")
    parser.add_argument("--baseline-count", type=int, default=4750)
    parser.add_argument("--seed", type=int, default=20261023)
    return parser.parse_args()


def make_background(pool_images: list[Path], count: int, folder: Path, rng: random.Random) -> list[Path]:
    """Object-free images: a 12-20 px corner patch upscaled, or the whole image blurred beyond recognition."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        image = Image.open(rng.choice(pool_images)).convert("RGB")
        if i % 2 == 0:
            size = rng.randint(12, 20)
            x = rng.choice((0, 64 - size))
            y = rng.choice((0, 64 - size))
            image = image.crop((x, y, x + size, y + size)).resize((64, 64), Image.BICUBIC)
        else:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(6, 10)))
        path = folder / f"background_{i:05d}.png"
        image.save(path)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training = training_images(args.data_dir)
    counts = Counter(c for _, c, _ in training)
    shares = {c: counts[c] / len(training) for c in CLASSES}
    used = {sha for _, _, sha in training}
    unseen = unseen_pool(args.pool_dir, used, {c: round(args.baseline_count * shares[c]) for c in CLASSES}, 20261022)
    unseen_paths, unseen_truth = [p for p, _ in unseen], [c for _, c in unseen]
    excluded = set(unseen_paths)

    train_paths, train_labels = [], []
    for c in CLASSES:
        files = sorted(f for f in (args.pool_dir / POOL_NAMES.get(c, c)).iterdir()
                       if f.suffix.lower() in (".png", ".jpg", ".jpeg") and f not in excluded)
        chosen = rng.sample(files, args.per_class)
        train_paths += chosen
        train_labels += [c] * len(chosen)
    background = make_background(train_paths, args.background, args.output_dir / "background", rng)
    held = args.background // 6
    train_paths += background[held:]
    train_labels += [BACKGROUND] * (len(background) - held)

    name, location = args.generated.split("=", 1)
    folder, pattern = location.rsplit(":", 1)
    gen_paths = sorted(Path(folder).glob(pattern))

    dino = Features(BACKBONE, device, 256, args.dino_cache)
    dino_x = [dino("labeller_train", train_paths), dino("labeller_background_held", background[:held]),
              dino("unseen_real", unseen_paths), dino(name, gen_paths)]
    del dino
    clip = ClipFeatures(device, args.clip_cache)
    clip_x = [clip("labeller_train", train_paths), clip("labeller_background_held", background[:held]),
              clip("unseen_real", unseen_paths), clip(name, gen_paths)]
    del clip
    x_train, x_background, x_unseen, x_gen = [torch.cat([d, c], 1).numpy() for d, c in zip(dino_x, clip_x)]

    model = LogisticRegression(max_iter=5000, C=1.0)
    model.fit(x_train, [LABELS.index(l) for l in train_labels])
    with (args.output_dir / "labeller.pkl").open("wb") as handle:
        pickle.dump({"model": model, "labels": LABELS, "features": [BACKBONE, "openai/clip-vit-large-patch14"]}, handle)

    unseen_pred = [LABELS[i] for i in model.predict(x_unseen)]
    background_pred = [LABELS[i] for i in model.predict(x_background)]
    matrix = confusion_matrix(unseen_truth, unseen_pred, labels=LABELS)
    gen_prob = model.predict_proba(x_gen)
    gen_labels = [LABELS[i] for i in gen_prob.argmax(1)]

    check = {
        "unseen_real_balanced_accuracy": float(balanced_accuracy_score(unseen_truth, unseen_pred)),
        "unseen_real_called_background": unseen_pred.count(BACKGROUND) / len(unseen_pred),
        "held_out_background_accuracy": background_pred.count(BACKGROUND) / len(background_pred),
        "unseen_real_confusion": {"labels": LABELS, "matrix": matrix.tolist()},
    }
    as_unrecognisable = ["unrecognisable" if l == BACKGROUND else l for l in gen_labels]
    result = metrics(as_unrecognisable, shares)
    (args.output_dir / "summary.json").write_text(json.dumps({"check": check, name: result}, indent=2) + "\n")
    with (args.output_dir / f"labels_{name}.csv").open("w", encoding="utf-8") as handle:
        handle.write("path,label,probability\n")
        handle.writelines(f"{p},{l},{q:.4f}\n" for p, l, q in zip(gen_paths, gen_labels, gen_prob.max(1)))
    for label in LABELS:
        members = [p for p, l in zip(gen_paths, gen_labels) if l == label][:120]
        if members:
            tiles = [torch.from_numpy(np.array(Image.open(p).convert("RGB"))).permute(2, 0, 1) / 255 for p in members]
            utils.save_image(torch.stack(tiles), args.output_dir / f"{name}_{label}.png", nrow=15, padding=2, pad_value=1)

    print(json.dumps({k: v for k, v in check.items() if k != "unseen_real_confusion"}, indent=2))
    print(f"{name}: background {result['unrecognisable_fraction']:.1%}, HDAI {result['HDAI']:.3f}, "
          f"LDAI {result['LDAI']:.3f}, log2DOR {result['log2_DOR']:+.3f}, TVD {result['TVD']:.3f}")
    print("amplification:", {c: round(v, 2) for c, v in result["class_amplification"].items()})
    print("counts:", dict(Counter(gen_labels)))


if __name__ == "__main__":
    main()
