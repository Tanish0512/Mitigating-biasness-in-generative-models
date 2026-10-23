"""Find which D0 classes and sources are blurry or hard to classify, and how G0 compares.

Every D0 image (real, or synthetic from the large DDPM: ``sample_*`` files) and
every G0 sample gets:

* sharpness: variance of the Laplacian of the grayscale 64x64 image (higher is
  sharper); compared against the real images of the same class;
* a DINOv2 linear-probe label, from a class-balanced probe trained on real D0
  images only (real images get 5-fold out-of-fold predictions, so no image is
  labelled by a probe that saw it);
* optionally a ResNet-18 label. The ResNet was trained on ``images/``, which
  contains most real D0 images, so its agreement on real images is optimistic.

A D0 image is flagged ``mislabel`` when both classifiers disagree with its
folder, and ``blurry`` when its sharpness is below the 5th percentile of real
images of its class. Flags are review candidates, not automatic removals.

Example::

    python diagnose_quality.py --d0-dir ../data/d0/dataset-0 \\
        --g0 "../data/d1/dataset-1:*/g0_sample_*.png" \\
        --resnet-checkpoint ../data/classifier/checkpoint_best.pt --output-dir ../results/diagnose
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from torchvision import transforms, utils
from tqdm.auto import tqdm

from evaluate_clip import CLASSES, load_resnet
from evaluate_features import BACKBONE, Features


LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d0-dir", type=Path, required=True)
    parser.add_argument("--g0", default=None, metavar="DIR:GLOB", help="G0 samples to compare")
    parser.add_argument("--resnet-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--blur-percentile", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def sharpness(paths: list[Path]) -> np.ndarray:
    values = []
    for path in tqdm(paths, desc="sharpness"):
        with Image.open(path) as image:
            gray = torch.from_numpy(np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32))
        values.append(float(F.conv2d(gray[None, None], LAPLACIAN).var()))
    return np.array(values)


@torch.inference_mode()
def resnet_labels(checkpoint: Path, paths: list[Path], batch_size: int, device: torch.device) -> list[str]:
    model, preprocess = load_resnet(checkpoint, device)
    labels = []
    for start in tqdm(range(0, len(paths), batch_size), desc="resnet"):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]])
        labels += [CLASSES[i] for i in model(batch.to(device)).argmax(1).tolist()]
    return labels


def save_grid(paths: list[str], path: Path, limit: int = 64) -> None:
    if paths:
        tensors = [transforms.ToTensor()(Image.open(p).convert("RGB").resize((64, 64))) for p in paths[:limit]]
        utils.save_image(torch.stack(tensors), path, nrow=8, padding=2)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract = Features(args.backbone, device, args.batch_size, args.output_dir / "cache")

    d0 = [(p, c, "synthetic" if p.name.startswith("sample_") else "real")
          for c in CLASSES for p in sorted((args.d0_dir / c).glob("*.png"))]
    paths = [p for p, _, _ in d0]
    folder = np.array([CLASSES.index(c) for _, c, _ in d0])
    is_real = np.array([s == "real" for _, _, s in d0])
    features = extract("d0", paths).numpy()
    sharp = sharpness(paths)

    probe = LogisticRegression(max_iter=3000, class_weight="balanced")
    folds = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    probe_index = np.empty(len(d0), dtype=int)
    probe_index[is_real] = cross_val_predict(probe, features[is_real], folder[is_real], cv=folds)
    probe.fit(features[is_real], folder[is_real])
    probe_index[~is_real] = probe.predict(features[~is_real])
    probe_label = [CLASSES[i] for i in probe_index]
    resnet = resnet_labels(args.resnet_checkpoint, paths, args.batch_size, device) if args.resnet_checkpoint else None

    thresholds = {c: float(np.percentile(sharp[is_real & (folder == i)], args.blur_percentile))
                  for i, c in enumerate(CLASSES)}
    rows = []
    for i, (path, cls, source) in enumerate(d0):
        probe_ok = probe_label[i] == cls
        resnet_ok = resnet[i] == cls if resnet else probe_ok
        rows.append({
            "path": str(path), "class": cls, "source": source, "sharpness": round(float(sharp[i]), 2),
            "probe_label": probe_label[i], "resnet_label": resnet[i] if resnet else "",
            "mislabel": bool(not probe_ok and not resnet_ok), "blurry": bool(sharp[i] < thresholds[cls]),
        })
    with (args.output_dir / "d0_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = {}
    grids = args.output_dir / "flagged"
    grids.mkdir(exist_ok=True)
    for cls in CLASSES:
        real_median = float(np.median([r["sharpness"] for r in rows if r["class"] == cls and r["source"] == "real"]))
        for source in ("real", "synthetic"):
            group = [r for r in rows if r["class"] == cls and r["source"] == source]
            table[f"{cls}/{source}"] = {
                "images": len(group),
                "median_sharpness": float(np.median([r["sharpness"] for r in group])),
                "sharpness_vs_real": float(np.median([r["sharpness"] for r in group])) / real_median,
                "probe_agrees": float(np.mean([r["probe_label"] == cls for r in group])),
                "resnet_agrees": float(np.mean([r["resnet_label"] == cls for r in group])) if resnet else None,
                "mislabel": sum(r["mislabel"] for r in group),
                "blurry": sum(r["blurry"] for r in group),
            }
            save_grid([r["path"] for r in group if r["mislabel"]], grids / f"{cls}_{source}_mislabel.png")

    g0_table = None
    if args.g0:
        folder_g0, pattern = args.g0.rsplit(":", 1)
        g0_paths = sorted(Path(folder_g0).glob(pattern))
        g0_features = extract("g0", g0_paths).numpy()
        g0_probs = probe.predict_proba(g0_features)
        g0_probe = [CLASSES[i] for i in g0_probs.argmax(1)]
        g0_resnet = resnet_labels(args.resnet_checkpoint, g0_paths, args.batch_size, device) if resnet else None
        g0_sharp = sharpness(g0_paths)
        g0_table = {}
        for i, cls in enumerate(CLASSES):
            members = [j for j, label in enumerate(g0_probe) if label == cls]
            real_median = table[f"{cls}/real"]["median_sharpness"]
            g0_table[cls] = {
                "images": len(members),
                "median_sharpness": float(np.median(g0_sharp[members])) if members else None,
                "sharpness_vs_real": float(np.median(g0_sharp[members])) / real_median if members else None,
                "probe_confidence": float(g0_probs[members, i].mean()) if members else None,
                "low_confidence_fraction": float((g0_probs[members, i] < 0.5).mean()) if members else None,
                "resnet_agrees": float(np.mean([g0_resnet[j] == cls for j in members])) if resnet and members else None,
            }
            order = sorted(members, key=lambda j: g0_sharp[j])
            save_grid([str(g0_paths[j]) for j in order], grids / f"g0_{cls}_blurriest.png")

    summary = {"backbone": args.backbone, "blur_thresholds": thresholds, "d0": table, "g0_by_probe_label": g0_table}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"{'D0 class/source':20s}{'n':>6s}{'sharp':>8s}{'vs real':>9s}{'probe ok':>10s}{'resnet ok':>11s}{'mislabel':>10s}{'blurry':>8s}")
    for key, r in table.items():
        resnet_ok = f"{r['resnet_agrees']:.1%}" if r["resnet_agrees"] is not None else "-"
        print(f"{key:20s}{r['images']:6d}{r['median_sharpness']:8.0f}{r['sharpness_vs_real']:9.2f}"
              f"{r['probe_agrees']:10.1%}{resnet_ok:>11s}{r['mislabel']:10d}{r['blurry']:8d}")
    if g0_table:
        print(f"\n{'G0 (probe label)':20s}{'n':>6s}{'sharp':>8s}{'vs real':>9s}{'conf':>8s}{'conf<0.5':>10s}{'resnet ok':>11s}")
        for cls, r in g0_table.items():
            if r["images"]:
                resnet_ok = f"{r['resnet_agrees']:.1%}" if r["resnet_agrees"] is not None else "-"
                print(f"{cls:20s}{r['images']:6d}{r['median_sharpness']:8.0f}{r['sharpness_vs_real']:9.2f}"
                      f"{r['probe_confidence']:8.2f}{r['low_confidence_fraction']:10.1%}{resnet_ok:>11s}")


if __name__ == "__main__":
    main()
