"""Measure higher-density bias of generated images in a DINOv2 feature space.

Real D0 images (the D0-sampled rows of D1's manifest, which keep their
true class folder) form the reference. They are split, stratified by class, into
a reference set and a held-out real "baseline generator" with D0's class ratios:
the baseline shows what an unbiased generator scores, so every metric can be
read relative to it instead of against an ideal that no labeller reaches.

For each generated set this reports:

* class shares from a class-balanced linear probe on DINOv2 features (no
  prior pull toward any class), with HDAI, TVD and JSD against D0;
* the same after dropping off-manifold images, i.e. images outside every real
  image's k-nearest-neighbour ball (Naeem et al., 2020);
* per-class coverage: the fraction of real images of a class that have a
  generated image inside their k-NN ball. Higher-density bias shows up as low
  coverage of low-density classes. CGI (coverage gap index) is the low-group
  mean coverage divided by the high-group mean coverage, normalised by the same
  ratio for the real baseline: 1 means no bias, below 1 means the low-density
  classes are under-covered.

Example::

    python evaluate_features.py --d1-dir ../data/d1/dataset-1 \\
        --generated G0=../data/d1/dataset-1:*/g0_sample_*.png \\
        --generated G1=../results/eval_g1/samples:*.png --output-dir ../results/features
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm.auto import tqdm

from evaluate_clip import CLASSES, D0_COUNTS, GROUPS, distribution_metrics


BACKBONE = "facebook/dinov2-base"
REFERENCE = {c: n / sum(D0_COUNTS.values()) for c, n in D0_COUNTS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d1-dir", type=Path, required=True, help="dataset-1 folder with manifest.csv")
    parser.add_argument("--generated", action="append", default=[], metavar="NAME=DIR:GLOB",
                        help="Generated image set to evaluate; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--baseline-count", type=int, default=4750, help="Held-out real images in the baseline")
    parser.add_argument("--k", type=int, default=5, help="Neighbours defining each real image's ball")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def real_images(d1_dir: Path) -> list[tuple[Path, str]]:
    with (d1_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # The shipped D1 manifest uses folder_class/D0_sample; build_dataset1.py writes class/dataset-0_sampled.
    class_key = "folder_class" if rows and "folder_class" in rows[0] else "class"
    rows = [r for r in rows if r["source"] in ("D0_sample", "dataset-0_sampled")]
    if not rows:
        raise ValueError(f"No D0 rows found in {d1_dir / 'manifest.csv'}")
    return [(d1_dir / r[class_key] / r["filename"], r[class_key]) for r in rows]


def split_real(images: list[tuple[Path, str]], count: int, seed: int):
    """Hold out ``count`` real images with D0's class ratios as the baseline generator."""
    rng = random.Random(seed)
    by_class: dict[str, list] = {c: [] for c in CLASSES}
    for item in images:
        by_class[item[1]].append(item)
    reference, baseline = [], []
    for c in CLASSES:
        items = by_class[c]
        rng.shuffle(items)
        take = round(count * REFERENCE[c])
        if take >= len(items):
            raise ValueError(f"Not enough real {c} images for a baseline of {count}")
        baseline += items[:take]
        reference += items[take:]
    return reference, baseline


class Features:
    def __init__(self, name: str, device: torch.device, batch_size: int, cache_dir: Path):
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.device, self.batch_size, self.cache_dir = device, batch_size, cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def __call__(self, name: str, paths: list[Path]) -> torch.Tensor:
        cache = self.cache_dir / f"{name}.pt"
        keys = [str(p) for p in paths]
        if cache.is_file():
            stored = torch.load(cache, weights_only=True)
            if stored["paths"] == keys:
                return stored["features"]
        chunks = []
        for start in tqdm(range(0, len(paths), self.batch_size), desc=f"features {name}"):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + self.batch_size]]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            cls = self.model(**inputs).last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu())
        features = torch.cat(chunks)
        torch.save({"paths": keys, "features": features}, cache)
        return features


def knn_radii(reference: torch.Tensor, k: int, device: torch.device) -> torch.Tensor:
    ref = reference.to(device)
    radii = []
    for start in range(0, len(ref), 2048):
        distances = torch.cdist(ref[start:start + 2048], ref)
        radii.append(distances.kthvalue(k + 1, dim=1).values)  # +1 skips the point itself
    return torch.cat(radii).cpu()


def manifold_stats(reference: torch.Tensor, radii: torch.Tensor, generated: torch.Tensor, device: torch.device):
    """Per generated image: inside any real ball? Per real image: covered by any generated image?"""
    ref, rad = reference.to(device), radii.to(device)
    covered = torch.zeros(len(ref), dtype=torch.bool, device=device)
    inside = []
    for start in range(0, len(generated), 2048):
        within = torch.cdist(generated[start:start + 2048].to(device), ref) <= rad[None, :]
        inside.append(within.any(1).cpu())
        covered |= within.any(0)
    return torch.cat(inside), covered.cpu()


def evaluate_set(name, features, probe, ref_features, ref_labels, radii, device) -> tuple[dict, list[str], torch.Tensor]:
    labels = [CLASSES[i] for i in probe.predict(features.numpy())]
    inside, covered = manifold_stats(ref_features, radii, features, device)
    coverage = {c: float(covered[[l == c for l in ref_labels]].float().mean()) for c in CLASSES}
    group_coverage = {g: float(np.mean([coverage[c] for c in members])) for g, members in GROUPS.items()}
    on_manifold = Counter(l for l, keep in zip(labels, inside.tolist()) if keep)
    return {
        "images": len(labels),
        "off_manifold_fraction": float(1 - inside.float().mean()),
        "probe_all": distribution_metrics(Counter(labels), REFERENCE),
        "probe_on_manifold": distribution_metrics(on_manifold, REFERENCE),
        "coverage": coverage,
        "group_coverage": group_coverage,
        "coverage_low_over_high": group_coverage["low"] / group_coverage["high"],
    }, labels, inside


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract = Features(args.backbone, device, args.batch_size, args.output_dir / "cache")

    reference, baseline = split_real(real_images(args.d1_dir), args.baseline_count, args.seed)
    ref_features = extract("real_reference", [p for p, _ in reference])
    ref_labels = [c for _, c in reference]
    base_features = extract("real_baseline", [p for p, _ in baseline])
    base_labels = [c for _, c in baseline]

    # Class-balanced probe: trained on the reference split, checked on the held-out baseline.
    probe = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    probe.fit(ref_features.numpy(), [CLASSES.index(c) for c in ref_labels])
    predicted = probe.predict(base_features.numpy())
    truth = [CLASSES.index(c) for c in base_labels]
    matrix = confusion_matrix(truth, predicted, labels=range(len(CLASSES)))
    probe_check = {
        "balanced_accuracy_on_real_baseline": float(balanced_accuracy_score(truth, predicted)),
        "recall_by_class": {c: float(matrix[i, i] / matrix[i].sum()) for i, c in enumerate(CLASSES)},
        "confusion_matrix": matrix.tolist(),
    }
    radii = knn_radii(ref_features, args.k, device)

    sets = {"real_baseline": base_features}
    for spec in args.generated:
        name, location = spec.split("=", 1)
        folder, pattern = location.rsplit(":", 1)
        paths = sorted(Path(folder).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No images for {name}: {folder}/{pattern}")
        sets[name] = extract(name, paths)
        with (args.output_dir / f"labels_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            handle.write("path\n" + "\n".join(map(str, paths)) + "\n")

    results = {}
    for name, features in sets.items():
        results[name], labels, inside = evaluate_set(name, features, probe, ref_features, ref_labels, radii, device)
        if name != "real_baseline":
            path_file = args.output_dir / f"labels_{name}.csv"
            paths = path_file.read_text(encoding="utf-8").splitlines()[1:]
            with path_file.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "probe_label", "on_manifold"])
                writer.writerows(zip(paths, labels, inside.tolist()))
    base_ratio = results["real_baseline"]["coverage_low_over_high"]
    for result in results.values():
        result["CGI"] = result["coverage_low_over_high"] / base_ratio

    summary = {
        "backbone": args.backbone,
        "k": args.k,
        "seed": args.seed,
        "reference_images": len(reference),
        "baseline_images": len(baseline),
        "probe_check": probe_check,
        "sets": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"probe balanced accuracy on held-out real images: {probe_check['balanced_accuracy_on_real_baseline']:.3f}")
    print(f"{'set':14s}{'HDAI':>7s}{'HDAI-on':>9s}{'TVD':>7s}{'JSD':>7s}{'off-man':>9s}{'cov H':>7s}{'cov M':>7s}{'cov L':>7s}{'CGI':>7s}")
    for name, r in results.items():
        g = r["group_coverage"]
        print(f"{name:14s}{r['probe_all']['HDAI']:7.3f}{r['probe_on_manifold']['HDAI']:9.3f}{r['probe_all']['TVD']:7.3f}"
              f"{r['probe_all']['JSD_bits']:7.3f}{r['off_manifold_fraction']:9.1%}{g['high']:7.3f}{g['medium']:7.3f}{g['low']:7.3f}{r['CGI']:7.3f}")


if __name__ == "__main__":
    main()
