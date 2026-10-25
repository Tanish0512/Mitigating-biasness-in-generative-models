"""Label a folder of generated images with labeller v2 (from train_labeller.py).

Writes ``image,label,probability,counted_label_p<threshold>`` like the label
files in results_g0_v2/, so the output can be passed to build_dataset1_v2.py.

Example::

    python apply_labeller.py --labeller results_g0_v2/classifier_new_p0.9/labeller.pkl \\
        --images ../results/samples_g0_v2_extra --cache ../results/labeller_cache --output labels.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import torch

from compare_labellers import ClipFeatures
from evaluate_features import BACKBONE, Features

BACKGROUND = "background"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labeller", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with args.labeller.open("rb") as handle:
        stored = pickle.load(handle)
    model, labels = stored["model"], stored["labels"]
    paths = sorted(args.images.glob("*.png"))
    name = args.images.name

    dino = Features(BACKBONE, device, 256, args.cache)
    dino_x = dino(name, paths)
    del dino
    clip_x = ClipFeatures(device, args.cache)(name, paths)
    probabilities = model.predict_proba(torch.cat([dino_x, clip_x], 1).numpy())

    key = f"counted_label_p{args.threshold}"
    rows = []
    for path, p in zip(paths, probabilities):
        label = labels[p.argmax()]
        counted = label if p.max() >= args.threshold and label != BACKGROUND else "unrecognisable"
        rows.append([path.name, label, f"{p.max():.4f}", counted])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "probability", key])
        writer.writerows(rows)
    print(dict(Counter(r[3] for r in rows)))


if __name__ == "__main__":
    main()
