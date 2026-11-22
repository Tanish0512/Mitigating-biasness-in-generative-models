"""Per-class representation and image quality of every model, in DINOv2 space.

For each class c and model (4,750 samples; labeller v2 counts at p >= threshold):

* representation: confident count and amplification A_c = q_c / p_c (p_c = D0 share);
* fidelity: precision_c, the share of samples counted as c that fall inside the k-NN ball of at least
  one real class-c image, and density_c, the mean number of real class-c balls containing such a
  sample divided by k (Naeem et al., 2020; ~1 for real images);
* diversity: coverage_c, the share of real class-c images whose ball contains at least one sample;
* pass rate: the share of samples the labeller assigns to c that reach the threshold.

Balls use k = 5 among all real reference images, as in evaluate_v2.py, so coverage matches it.
The real held-out D0 split is scored the same way as a reference "ideal generator".

Example::

    python src/evaluation/per_class_quality.py --output-dir ../results/per_class_quality \\
        --model "G0=results/g0_v2/classifier_new_p0.9/labels_G0v2.csv:../results/eval_all6/cache/G0v2.pt"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from evaluate_features import knn_radii
from evaluate_v2 import CLASSES

REFERENCE = Path("../results/eval_all6/cache/train_reference.pt")
BASELINE = Path("../results/eval_all6/cache/train_baseline.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=LABELS:DINO_CACHE")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--images", type=int, default=4750)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True)


def score(features: torch.Tensor, counted: list[str], ref: torch.Tensor, ref_class: torch.Tensor,
          radii: torch.Tensor, k: int) -> dict:
    inside = torch.cdist(features, ref) <= radii[None, :]
    covered = inside.any(0)
    out = {}
    for c_index, c in enumerate(CLASSES):
        real_c = ref_class == c_index
        rows = torch.tensor([label == c for label in counted])
        within = inside[rows][:, real_c]
        out[c] = {
            "count": int(rows.sum()),
            "precision": round(within.any(1).float().mean().item(), 3) if rows.any() else None,
            "density": round(within.sum(1).float().mean().item() / k, 3) if rows.any() else None,
            "coverage": round(covered[real_c].float().mean().item(), 3),
        }
    return out


def main() -> None:
    args = parse_args()
    reference = torch.load(REFERENCE, weights_only=True)
    ref = unit(reference["features"].float())
    ref_class = torch.tensor([CLASSES.index(Path(p).parent.name) for p in reference["paths"]])
    radii = knn_radii(ref, args.k, torch.device("cpu"))
    shares = {c: (ref_class == i).float().mean().item() for i, c in enumerate(CLASSES)}

    results = {}
    baseline = torch.load(BASELINE, weights_only=True)
    results["real held-out"] = score(unit(baseline["features"].float()),
                                     [Path(p).parent.name for p in baseline["paths"]], ref, ref_class, radii, args.k)
    for spec in args.model:
        name, rest = spec.split("=", 1)
        labels_path, cache = rest.split(":")
        with open(labels_path, newline="") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda r: r["image"])[:args.images]
        stored = torch.load(cache, weights_only=True)
        index = {Path(p).name: i for i, p in enumerate(stored["paths"])}
        features = unit(stored["features"][[index[r["image"]] for r in rows]].float())
        counted = [r["label"] if float(r["probability"]) >= args.threshold and r["label"] in CLASSES else "none"
                   for r in rows]
        scores = score(features, counted, ref, ref_class, radii, args.k)
        total = sum(scores[c]["count"] for c in CLASSES)
        for c in CLASSES:
            predicted = sum(r["label"] == c for r in rows)
            scores[c]["amplification"] = round(scores[c]["count"] / total / shares[c], 3)
            scores[c]["pass_rate"] = round(scores[c]["count"] / predicted, 3) if predicted else None
        results[name] = scores
        print(f"{name} done", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_class_quality.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for metric in ("amplification", "precision", "density", "coverage"):
        print(f"\n{metric}")
        print(f"{'model':14s}" + "".join(f"{c[:8]:>9s}" for c in CLASSES))
        for name, scores in results.items():
            cells = [scores[c].get(metric) for c in CLASSES]
            print(f"{name:14s}" + "".join(f"{'-':>9s}" if v is None else f"{v:9.2f}" for v in cells))


if __name__ == "__main__":
    main()
