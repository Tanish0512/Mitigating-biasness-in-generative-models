"""Audit the images labeller v2 assigns to one class, with checks that do not use labeller v2.

For every model, the images counted as ``--class-name`` (labeller v2, p >= threshold) are checked by:

* a DINOv2-only probe: class-balanced logistic regression on real D0 reference images
  (no CLIP features, different training images from labeller v2);
* CLIP zero-shot: the six class prompts plus "unrecognisable" prompts, no training;
* a k-NN vote among the 10 nearest real D0 reference images in DINOv2 space;
* the real manifolds: whether the image falls inside the k-NN ball (k = 5) of at least one real
  image of the audited class, and of at least one real ship / other image.

A grid of random counted images per model is saved for visual inspection. The DINOv2 features
come from evaluate_v2.py caches; CLIP image features from apply_labeller.py caches or, for the
few audited images, are computed here.

Example::

    python src/evaluation/class_audit.py --class-name marine_animal --output-dir ../results/marine_audit \\
        --model "G0=results/g0_v2/classifier_new_p0.9/labels_G0v2.csv:../results/samples_g0_v2:../results/eval_all6/cache/G0v2.pt"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import evaluate_clip
from compare_labellers import CLASSES, CLIP_NAMES

REFERENCE = Path("../results/eval_all6/cache/train_reference.pt")
CLIP_CACHES = [Path("../results/labeller_cache"), Path("../results/labellers_g0_v2/cache"),
               Path("../results/labeller_v2/cache")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-name", default="marine_animal")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=LABELS:SAMPLES:DINO_CACHE")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=4750)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261120)
    return parser.parse_args()


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True)


def clip_text(model, processor) -> torch.Tensor:
    def embed(prompts):
        tokens = processor(text=prompts, return_tensors="pt", padding=True)
        f = evaluate_clip._projected(model.get_text_features(**tokens))
        mean = unit(f).mean(0)
        return mean / mean.norm()

    with torch.inference_mode():
        rows = [embed([t.format(n) for n in CLIP_NAMES[c] for t in evaluate_clip.TEMPLATES]) for c in CLASSES]
        rows.append(embed(evaluate_clip.UNRECOGNISABLE_PROMPTS))
    return torch.stack(rows).float()


def cached_clip(samples: Path, paths: list[str]) -> dict[str, torch.Tensor]:
    for folder in CLIP_CACHES:
        cache = folder / f"clip_{samples.name}.pt"
        if cache.is_file():
            stored = torch.load(cache, weights_only=True)
            index = {Path(p).name: i for i, p in enumerate(stored["paths"])}
            if all(Path(p).name in index for p in paths):
                return {p: stored["features"][index[Path(p).name]].float() for p in paths}
    return {}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    target = CLASSES.index(args.class_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = torch.load(REFERENCE, weights_only=True)
    ref_x = unit(reference["features"].float())
    ref_y = np.array([CLASSES.index(Path(p).parent.name) for p in reference["paths"]])
    probe = LogisticRegression(max_iter=3000, class_weight="balanced").fit(ref_x.numpy(), ref_y)
    # k-NN radius (k = 5) of every real reference image within its own class.
    radius = torch.empty(len(ref_x))
    for c in range(len(CLASSES)):
        members = torch.from_numpy(np.where(ref_y == c)[0])
        d = torch.cdist(ref_x[members], ref_x[members])
        radius[members] = d.kthvalue(6, dim=1).values

    model, processor, _ = evaluate_clip.load_clip(evaluate_clip.CLIP_MODEL, torch.device("cpu"))
    text = clip_text(model, processor)
    results = {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        labels_path, samples, dino_cache = rest.split(":")
        samples = Path(samples)
        with open(labels_path, newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda r: r["image"])[:args.images]
        hits = [r["image"] for r in rows if r["label"] == args.class_name and float(r["probability"]) >= args.threshold]
        stored = torch.load(dino_cache, weights_only=True)
        index = {Path(p).name: i for i, p in enumerate(stored["paths"])}
        x = unit(stored["features"][[index[h] for h in hits]].float())

        probe_pred = probe.predict(x.numpy())
        distance = torch.cdist(x, ref_x)
        knn = ref_y[distance.topk(10, largest=False).indices.numpy()]
        knn_vote = np.array([np.bincount(row, minlength=len(CLASSES)).argmax() for row in knn])
        inside = distance <= radius[None, :]
        on_class = inside[:, torch.from_numpy(ref_y == target)].any(1)
        on_ship = inside[:, torch.from_numpy(ref_y == CLASSES.index("ship"))].any(1)
        on_any = inside.any(1)

        paths = [str(samples / h) for h in hits]
        clip_x = cached_clip(samples, paths)
        missing = [p for p in paths if p not in clip_x]
        if missing:
            with torch.inference_mode():
                for begin in range(0, len(missing), 32):
                    batch = [Image.open(p).convert("RGB") for p in missing[begin:begin + 32]]
                    pixels = processor(images=batch, return_tensors="pt")
                    features = evaluate_clip._projected(model.get_image_features(**pixels)).float()
                    clip_x.update(zip(missing[begin:begin + 32], features))
        clip_pred = (unit(torch.stack([clip_x[p] for p in paths])) @ text.T).argmax(1).numpy()

        def share(pred):
            counts = Counter(CLASSES[i] if i < len(CLASSES) else "unrecognisable" for i in pred)
            return {"agree": round(counts[args.class_name] / max(len(pred), 1), 3),
                    "ship": round(counts["ship"] / max(len(pred), 1), 3), "counts": dict(counts)}

        results[name] = {
            "counted": len(hits),
            "dino_probe": share(probe_pred),
            "clip_zero_shot": share(clip_pred),
            "knn_vote": share(knn_vote),
            "on_real_class_manifold": round(on_class.float().mean().item(), 3) if hits else None,
            "on_real_ship_manifold": round(on_ship.float().mean().item(), 3) if hits else None,
            "off_all_real_manifolds": round(1 - on_any.float().mean().item(), 3) if hits else None,
            "all_three_checks_agree": round(float(np.mean((probe_pred == target) & (clip_pred == target)
                                                          & (knn_vote == target))), 3) if hits else None,
        }
        chosen = rng.sample(hits, min(args.grid, len(hits)))
        grid = Image.new("RGB", (10 * 66, ((len(chosen) + 9) // 10) * 66), "white")
        for i, h in enumerate(chosen):
            grid.paste(Image.open(samples / h).convert("RGB").resize((64, 64)), ((i % 10) * 66, (i // 10) * 66))
        grid.save(args.output_dir / f"grid_{name}.png")

    (args.output_dir / "audit.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"{'model':10s}{'counted':>8s}{'DINO':>7s}{'CLIP':>7s}{'kNN':>7s}{'all 3':>7s}"
          f"{'on cls':>8s}{'on ship':>8s}{'off all':>8s}   (agreement with labeller v2)")
    for name, r in results.items():
        if r["counted"]:
            print(f"{name:10s}{r['counted']:8d}{r['dino_probe']['agree']:7.2f}{r['clip_zero_shot']['agree']:7.2f}"
                  f"{r['knn_vote']['agree']:7.2f}{r['all_three_checks_agree']:7.2f}{r['on_real_class_manifold']:8.2f}"
                  f"{r['on_real_ship_manifold']:8.2f}{r['off_all_real_manifolds']:8.2f}")


if __name__ == "__main__":
    main()
