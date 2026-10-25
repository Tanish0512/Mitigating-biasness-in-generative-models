"""Build dataset-1-v2 (D1) for G1: 75% of dataset-0-v2 plus 25% unfiltered G0 v2 samples.

Follows the original D1 design (replacement fraction r, total size fixed):

* ``(1 - r)`` of every dataset-0-v2 class is kept, sampled at random within the
  class, so the kept part has D0's 20:1 ratio;
* ``r * total`` G0 samples, chosen at random from the generated set, are added
  without any filtering. Their folder is the labeller v2 label at p >= 0.9
  (``unrecognisable`` below that); G1 is unconditional, so folders only record
  the composition.

``--curate`` (mitigation) replaces the random, unfiltered choice: only G0
samples whose labeller v2 probability is at least ``--threshold`` are used, and
each class gets r * total * (its dataset-0-v2 share) of them, so the G0 part has
D0's class ratios and no failed images. The run stops and reports the shortfall
when a class has too few confident samples.

``manifest.csv`` lists every image with its folder, source (``d0v2_real``,
``d0v2_synthetic`` or ``g0v2``), source path and, for G0 images, the labeller's
label and probability.

Example::

    python build_dataset1_v2.py --d0-dir ../data/dataset-0-v2 --g0-samples ../results/samples_g0_v2 \\
        --labels results_g0_v2/classifier_new_p0.9/labels_G0v2.csv --output-dir ../data/dataset-1-v2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

LABEL_KEY = "counted_label_p0.9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d0-dir", type=Path, required=True)
    parser.add_argument("--g0-samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="labels_G0v2.csv from labeller v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r", type=float, default=0.25, help="Fraction of D1 replaced by G0 samples")
    parser.add_argument("--g0-checkpoint", default="runs/g0_v2/final.pt", help="Recorded in summary.json")
    parser.add_argument("--curate", action="store_true", help="Confident G0 samples in D0's class ratios")
    parser.add_argument("--threshold", type=float, default=0.9, help="Minimum labeller probability with --curate")
    parser.add_argument("--seed", type=int, default=20261023)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists")

    with (args.d0_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        d0 = list(csv.DictReader(handle))
    total = len(d0)
    rows, kept = [], Counter()
    for name in sorted({r["class"] for r in d0}):
        members = [r for r in d0 if r["class"] == name]
        for r in rng.sample(members, round(len(members) * (1 - args.r))):
            source = "d0v2_synthetic" if r["source"].startswith("synthetic") else "d0v2_real"
            rows.append((name, r["filename"], source, f"dataset-0-v2/{name}/{r['filename']}", "", ""))
            kept[name] += 1

    with args.labels.open(newline="", encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    needed = total - len(rows)
    if args.curate:
        d0_counts = Counter(r["class"] for r in d0)
        exact = {c: needed * n / total for c, n in d0_counts.items()}
        quota = {c: int(v) for c, v in exact.items()}
        for c in sorted(exact, key=lambda c: exact[c] - quota[c], reverse=True)[:needed - sum(quota.values())]:
            quota[c] += 1
        pools = {c: [r for r in labels if r["label"] == c and float(r["probability"]) >= args.threshold]
                 for c in quota}
        short = {c: f"{len(pools[c])}/{quota[c]}" for c in quota if len(pools[c]) < quota[c]}
        if short:
            raise SystemExit(f"Too few confident G0 samples (have/need): {short}")
        chosen = [r for c in sorted(quota) for r in rng.sample(pools[c], quota[c])]
        selection = f"curated: labeller v2 probability >= {args.threshold}, per-class quotas {quota}"
    else:
        chosen = rng.sample(labels, needed)
        selection = "random, unfiltered; folder = labeller v2 label at p >= 0.9, else unrecognisable"
    for r in chosen:
        folder = r["label"] if args.curate else r[LABEL_KEY]
        rows.append((folder, r["image"], "g0v2", f"{args.g0_samples.name}/{r['image']}", r["label"], r["probability"]))

    for folder, filename, source, origin, _, _ in rows:
        target = args.output_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        original = args.g0_samples / filename if source == "g0v2" else args.d0_dir / origin.split("/", 1)[1]
        shutil.copy2(original, target / filename)

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["folder", "filename", "source", "source_path", "labeller_label", "labeller_probability"])
        writer.writerows(rows)
    by_folder = Counter(r[0] for r in rows)
    by_source = Counter((r[0], r[2]) for r in rows)
    summary = {
        "generation": "D1 v2 for training G1 v2" + (" (curated mixing)" if args.curate else ""),
        "source_dataset": "dataset-0-v2",
        "g0_checkpoint": args.g0_checkpoint,
        "replacement_fraction_r": args.r,
        "seed": args.seed,
        "total": len(rows),
        "kept_from_d0v2": dict(kept),
        "g0_samples": total - sum(kept.values()),
        "g0_selection": selection,
        "folder_counts": dict(sorted(by_folder.items())),
        "folder_by_source": {f: {s: n for (ff, s), n in by_source.items() if ff == f} for f in sorted(by_folder)},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["folder_by_source"], indent=1))
    print("total", len(rows))


if __name__ == "__main__":
    main()
