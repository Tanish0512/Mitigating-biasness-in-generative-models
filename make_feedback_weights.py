"""Feedback-calibrated class weights from a generator's measured class amplification.

Each generation's per-class weight is corrected by how far that generator's
output share missed the target (D0) share:

    w_c <- w_prev_c * (target_c / generated_c) ** eta,  i.e. w_prev_c * A_c ** -eta

where A_c = generated share / target share among confidently labelled images
(labeller v2, ``counted_label_p0.9``). The weights are clipped to [low, high] and
then normalised so the mean weight over the training set is 1, as in
train_diffusion.py. The output JSON is passed to
``train_diffusion.py --class-weighting file --class-weights-file``.

Example::

    python make_feedback_weights.py --labels results_g0_v2/classifier_new_p0.9/labels_G0v2.csv \\
        --data-dir ../data/dataset-0-v2 --output ../runs/feedback/g0_fb_weights.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="Label CSV of the generator being corrected")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Training set the weights are for (class counts for normalisation)")
    parser.add_argument("--target-dir", type=Path, default=None,
                        help="Dataset whose class shares are the target (default: --data-dir)")
    parser.add_argument("--previous", type=Path, default=None, help="Previous generation's weights JSON")
    parser.add_argument("--column", default="counted_label_p0.9")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--low", type=float, default=0.7)
    parser.add_argument("--high", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def class_counts(data_dir: Path) -> dict[str, int]:
    return {folder.name: sum(f.suffix.lower() in IMAGE_SUFFIXES for f in folder.iterdir())
            for folder in sorted(p for p in data_dir.iterdir() if p.is_dir())}


def main() -> None:
    args = parse_args()
    counts = class_counts(args.data_dir)
    target_counts = class_counts(args.target_dir or args.data_dir)
    target = {c: n / sum(target_counts.values()) for c, n in target_counts.items()}
    with args.labels.open(newline="", encoding="utf-8") as handle:
        labels = Counter(row[args.column] for row in csv.DictReader(handle))
    counted = sum(labels[c] for c in counts)
    # A class the generator never drew confidently gets the maximum weight via the clip.
    amplification = {c: labels[c] / counted / target[c] for c in counts}
    previous = json.loads(args.previous.read_text(encoding="utf-8")) if args.previous else {c: 1.0 for c in counts}

    raw = {c: previous[c] * (amplification[c] ** -args.eta if amplification[c] > 0 else float("inf")) for c in counts}
    clipped = {c: min(args.high, max(args.low, w)) for c, w in raw.items()}
    total = sum(counts.values())
    mean = sum(counts[c] / total * clipped[c] for c in counts)
    weights = {c: round(clipped[c] / mean, 4) for c in counts}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    record = {"labels": str(args.labels), "data_dir": str(args.data_dir), "target_dir": str(args.target_dir or args.data_dir),
              "previous": str(args.previous) if args.previous else None, "eta": args.eta, "clip": [args.low, args.high],
              "counted_images": counted, "amplification": {c: round(a, 4) for c, a in amplification.items()},
              "weights": weights}
    args.output.with_name(args.output.stem + "_record.json").write_text(json.dumps(record, indent=2) + "\n",
                                                                       encoding="utf-8")
    print(f"{'class':15s}{'A_c':>8s}{'weight':>9s}")
    for c in counts:
        print(f"{c:15s}{amplification[c]:8.3f}{weights[c]:9.3f}")


if __name__ == "__main__":
    main()
