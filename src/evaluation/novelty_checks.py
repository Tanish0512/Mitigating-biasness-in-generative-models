"""Checks that need no training: bootstrap intervals and the confidence filter's class bias.

1. Bootstrap: each model's first ``--images`` labelled samples are resampled
   with replacement; the 95% interval of log2 DOR and of every class's
   amplification shows which differences between models are larger than
   sampling noise. Metrics follow the report: confident labels only
   (labeller v2, p >= 0.9), D0 shares as the target.
2. Filter bias: for every class the labeller predicts (its argmax, before any
   threshold), the fraction of samples that pass p >= threshold, for thresholds
   0.6 to 0.9. A lower pass rate for rare classes means curation removes them
   more often.

Example::

    python src/evaluation/novelty_checks.py --data-dir ../data/dataset-0-v2 --output-dir ../results/novelty_checks \\
        --model G0=results/g0_v2/classifier_new_p0.9/labels_G0v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HIGH = ("aircraft", "car", "ship")
LOW = ("butterfly", "marine_animal", "human")
CLASSES = HIGH + LOW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="D0, for the target class shares")
    parser.add_argument("--model", action="append", required=True, help="NAME=labels.csv (repeatable)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=4750)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261116)
    return parser.parse_args()


def metrics(counts: np.ndarray, target: np.ndarray) -> np.ndarray:
    """log2 DOR and per-class amplification for each row of class counts (rows x classes)."""
    shares = counts / counts.sum(1, keepdims=True)
    amplification = shares / target
    high, low = counts[:, :3].sum(1), counts[:, 3:].sum(1)
    target_odds = target[:3].sum() / target[3:].sum()
    with np.errstate(divide="ignore"):
        log_dor = np.log2(high / low / target_odds)
    return np.column_stack([log_dor, amplification])


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n = {c: sum(1 for f in (args.data_dir / c).iterdir() if f.suffix.lower() in (".png", ".jpg")) for c in CLASSES}
    target = np.array([n[c] / sum(n.values()) for c in CLASSES])
    columns = ["log2_DOR", *[f"A_{c}" for c in CLASSES]]
    bootstrap, filter_rates = {}, {}

    for spec in args.model:
        name, path = spec.split("=", 1)
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))[:args.images]
        labels = np.array([r["label"] for r in rows])
        probability = np.array([float(r["probability"]) for r in rows])
        counted = np.where(probability >= args.threshold, labels, "none")
        # One-hot class membership of each counted image; resample image indices.
        onehot = np.stack([counted == c for c in CLASSES], 1).astype(np.int64)
        point = metrics(onehot.sum(0, keepdims=True), target)[0]
        draws = rng.integers(0, len(rows), size=(args.resamples, len(rows)))
        samples = metrics(onehot[draws].sum(1).astype(float), target)
        low, high = np.percentile(samples, [2.5, 97.5], axis=0)
        bootstrap[name] = {col: {"value": round(float(p), 3), "ci95": [round(float(a), 3), round(float(b), 3)]}
                           for col, p, a, b in zip(columns, point, low, high)}
        filter_rates[name] = {
            c: {str(t): round(float((probability[labels == c] >= t).mean()), 3) if (labels == c).any() else None
                for t in (0.6, 0.7, 0.8, 0.9)} | {"predicted": int((labels == c).sum())}
            for c in CLASSES}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "filter_pass_rates.json").write_text(json.dumps(filter_rates, indent=2) + "\n", encoding="utf-8")

    print(f"95% bootstrap intervals ({args.resamples} resamples of {args.images} images, p >= {args.threshold})")
    print(f"{'model':12s}" + "".join(f"{c:>22s}" for c in columns[:1] + ["A_butterfly", "A_marine_animal", "A_human"]))
    for name, values in bootstrap.items():
        cells = []
        for col in ("log2_DOR", "A_butterfly", "A_marine_animal", "A_human"):
            v = values[col]
            cells.append(f"{v['value']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}]" if col == "log2_DOR"
                         else f"{v['value']:.2f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]")
        print(f"{name:12s}" + "".join(f"{c:>22s}" for c in cells))
    print("\nShare of predicted images that pass p >= 0.9 (filter pass rate), by predicted class")
    print(f"{'model':12s}" + "".join(f"{c:>15s}" for c in CLASSES))
    for name, rates in filter_rates.items():
        print(f"{name:12s}" + "".join(f"{rates[c]['0.9']:>15.3f}" if rates[c]["0.9"] is not None else f"{'-':>15s}"
                                      for c in CLASSES))


if __name__ == "__main__":
    main()
