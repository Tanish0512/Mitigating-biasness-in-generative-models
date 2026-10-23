"""Measure higher-density bias and copying of generated images against dataset-0-v2.

The reference distribution is the model's own training set (dataset-0-v2, real
and synthetic, 6 classes in a high/low 20:1 mixture). It is split, stratified by
class, into a reference set and a held-out "baseline generator" with the
training class ratios, so every metric can be read against what an unbiased
generator scores. Metrics, in DINOv2 feature space (see evaluate_features.py):

* class shares from a class-balanced linear probe, with HDAI, TVD and JSD;
* the same for on-manifold images only, and the off-manifold fraction;
* per-class coverage (k-NN balls, Naeem et al., 2020) and CGI, the low/high
  coverage ratio normalised by the baseline's (1 = no bias);
* copying: each image's nearest training image by DINOv2 cosine similarity and by
  pixel distance at 64x64. Real pool images never used in training give the
  closeness that is normal for a fresh image of the same classes.

Example::

    python evaluate_v2.py --data-dir ../data/dataset-0-v2 --pool-dir ../data/pool \\
        --generated G0=../results/samples_g0_v2:*.png --output-dir ../results/eval_g0_v2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchvision import utils
from tqdm.auto import tqdm

from evaluate_features import BACKBONE, Features, knn_radii, manifold_stats

CLASSES = ["aircraft", "car", "ship", "butterfly", "marine_animal", "human"]
GROUPS = {"high": ["aircraft", "car", "ship"], "low": ["butterfly", "marine_animal", "human"]}
POOL_NAMES = {"marine_animal": "fish"}
COPY_SIMILARITY = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="dataset-0-v2 folder with manifest.csv")
    parser.add_argument("--pool-dir", type=Path, required=True, help="100k image pool, for unseen real images")
    parser.add_argument("--generated", action="append", default=[], metavar="NAME=DIR:GLOB")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default=BACKBONE)
    parser.add_argument("--baseline-count", type=int, default=4750)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def distribution_metrics(counts: Counter, p: dict[str, float]) -> dict:
    total = sum(counts[c] for c in CLASSES)
    q = {c: counts[c] / total for c in CLASSES}
    mix = {c: (p[c] + q[c]) / 2 for c in CLASSES}
    kl = lambda a, b: sum(a[c] * math.log2(a[c] / b[c]) for c in CLASSES if a[c] > 0)
    groups = {g: sum(q[c] for c in m) / sum(p[c] for c in m) for g, m in GROUPS.items()}
    return {
        "count": total,
        "class_counts": {c: counts[c] for c in CLASSES},
        "class_shares": q,
        "class_amplification": {c: q[c] / p[c] for c in CLASSES},
        "group_shares": {g: sum(q[c] for c in m) for g, m in GROUPS.items()},
        "group_amplification": groups,
        "HDAI": groups["high"],
        "TVD": 0.5 * sum(abs(p[c] - q[c]) for c in CLASSES),
        "JSD_bits": 0.5 * kl(p, mix) + 0.5 * kl(q, mix),
    }


def training_images(data_dir: Path) -> list[tuple[Path, str, str]]:
    with (data_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return [(data_dir / r["class"] / r["filename"], r["class"], r["sha256"]) for r in csv.DictReader(handle)]


def split(images, count: int, shares: dict[str, float], seed: int):
    rng = random.Random(seed)
    reference, baseline = [], []
    for c in CLASSES:
        items = [item for item in images if item[1] == c]
        rng.shuffle(items)
        take = round(count * shares[c])
        baseline += items[:take]
        reference += items[take:]
    return reference, baseline


def unseen_pool(pool_dir: Path, used: set[str], per_class: dict[str, int], seed: int) -> list[tuple[Path, str]]:
    """Pool images whose bytes are not in the training set, ``per_class[c]`` of each class."""
    rng = random.Random(seed)
    chosen = []
    for c in CLASSES:
        files = sorted((pool_dir / POOL_NAMES.get(c, c)).iterdir())
        rng.shuffle(files)
        picked = []
        for f in files:
            if len(picked) == per_class[c]:
                break
            if f.suffix.lower() in (".png", ".jpg", ".jpeg") and hashlib.sha256(f.read_bytes()).hexdigest() not in used:
                picked.append((f, c))
        chosen += picked
    return chosen


def pixels(paths: list[Path]) -> torch.Tensor:
    arrays = [np.asarray(Image.open(p).convert("RGB").resize((64, 64), Image.BICUBIC), dtype=np.float32) / 255
              for p in tqdm(paths, desc="pixels", leave=False)]
    return torch.from_numpy(np.stack(arrays)).flatten(1)


def nearest(queries: torch.Tensor, keys: torch.Tensor, device: torch.device, similarity: bool):
    """Index and value of each query's nearest key (max cosine similarity or min RMS pixel distance)."""
    keys = keys.to(device)
    values, indices = [], []
    for start in range(0, len(queries), 1024):
        q = queries[start:start + 1024].to(device)
        if similarity:
            v, i = (q @ keys.T).max(1)
        else:
            v, i = (torch.cdist(q, keys) / math.sqrt(keys.shape[1])).min(1)
        values.append(v.cpu())
        indices.append(i.cpu())
    return torch.cat(values), torch.cat(indices)


def copy_stats(name, paths, feats, pix, train_paths, train_feats, train_pix, device, output_dir) -> dict:
    sim, sim_index = nearest(feats, train_feats, device, similarity=True)
    dist, _ = nearest(pix, train_pix, device, similarity=False)
    with (output_dir / f"nearest_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "nearest_training_image", "dino_similarity", "pixel_rms_distance"])
        writer.writerows(zip(map(str, paths), (str(train_paths[i]) for i in sim_index), sim.tolist(), dist.tolist()))
    # Most similar pairs, generated above its nearest training image.
    order = sim.argsort(descending=True)[:16]
    load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((64, 64)))).permute(2, 0, 1) / 255
    tiles = [load(p) for i in order for p in (paths[i], train_paths[sim_index[i]])]
    utils.save_image(torch.stack(tiles), output_dir / f"nearest_pairs_{name}.png", nrow=8, padding=2, pad_value=1)
    return {
        "dino_similarity_median": float(sim.median()),
        "dino_similarity_p99": float(sim.quantile(0.99)),
        f"fraction_similarity_above_{COPY_SIMILARITY}": float((sim > COPY_SIMILARITY).float().mean()),
        "pixel_rms_distance_median": float(dist.median()),
        "pixel_rms_distance_p01": float(dist.quantile(0.01)),
        "distinct_nearest_training_images": int(sim_index.unique().numel()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extract = Features(args.backbone, device, args.batch_size, args.output_dir / "cache")

    training = training_images(args.data_dir)
    class_counts = Counter(c for _, c, _ in training)
    shares = {c: class_counts[c] / len(training) for c in CLASSES}
    reference, baseline = split(training, args.baseline_count, shares, args.seed)
    ref_features = extract("train_reference", [p for p, _, _ in reference])
    ref_labels = [c for _, c, _ in reference]
    base_features = extract("train_baseline", [p for p, _, _ in baseline])
    base_labels = [c for _, c, _ in baseline]
    train_paths = [p for p, _, _ in reference + baseline]
    train_feats = torch.cat([ref_features, base_features])
    train_pix = pixels(train_paths)

    probe = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    probe.fit(ref_features.numpy(), [CLASSES.index(c) for c in ref_labels])
    predicted = probe.predict(base_features.numpy())
    truth = [CLASSES.index(c) for c in base_labels]
    matrix = confusion_matrix(truth, predicted, labels=range(len(CLASSES)))
    probe_check = {
        "balanced_accuracy_on_baseline": float(balanced_accuracy_score(truth, predicted)),
        "recall_by_class": {c: float(matrix[i, i] / matrix[i].sum()) for i, c in enumerate(CLASSES)},
        "confusion_matrix": matrix.tolist(),
    }
    radii = knn_radii(ref_features, args.k, device)

    # Unseen real images with the training class ratios: closeness to the training set that is not copying.
    used = {sha for _, _, sha in training}
    unseen = unseen_pool(args.pool_dir, used, {c: round(args.baseline_count * shares[c]) for c in CLASSES}, args.seed)
    sets = {"unseen_real": ([p for p, _ in unseen], extract("unseen_real", [p for p, _ in unseen]))}
    for spec in args.generated:
        name, location = spec.split("=", 1)
        folder, pattern = location.rsplit(":", 1)
        paths = sorted(Path(folder).glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No images for {name}: {folder}/{pattern}")
        sets[name] = (paths, extract(name, paths))

    results = {"train_baseline": {}}
    evaluated = {"train_baseline": base_features, **{n: f for n, (_, f) in sets.items()}}
    for name, features in evaluated.items():
        labels = [CLASSES[i] for i in probe.predict(features.numpy())]
        inside, covered = manifold_stats(ref_features, radii, features, device)
        coverage = {c: float(covered[[l == c for l in ref_labels]].float().mean()) for c in CLASSES}
        group_coverage = {g: float(np.mean([coverage[c] for c in m])) for g, m in GROUPS.items()}
        results[name] = {
            "images": len(labels),
            "off_manifold_fraction": float(1 - inside.float().mean()),
            "probe_all": distribution_metrics(Counter(labels), shares),
            "probe_on_manifold": distribution_metrics(Counter(l for l, k in zip(labels, inside.tolist()) if k), shares),
            "coverage": coverage,
            "group_coverage": group_coverage,
            "coverage_low_over_high": group_coverage["low"] / group_coverage["high"],
        }
        if name in sets:
            paths = sets[name][0]
            results[name]["copying"] = copy_stats(name, paths, features, pixels(paths), train_paths, train_feats,
                                                  train_pix, device, args.output_dir)
            with (args.output_dir / f"labels_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["path", "probe_label", "on_manifold"])
                writer.writerows(zip(map(str, paths), labels, inside.tolist()))
            if name != "unseen_real":
                grid = []
                for c in CLASSES:
                    members = [p for p, l in zip(paths, labels) if l == c][:16]
                    grid += [utils.make_grid(torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB")))
                                                          .permute(2, 0, 1) / 255 for p in members]), nrow=16)
                             if members else None]
                for c, image in zip(CLASSES, grid):
                    if image is not None:
                        utils.save_image(image, args.output_dir / f"by_class_{name}_{c}.png")
    base_ratio = results["train_baseline"]["coverage_low_over_high"]
    for result in results.values():
        result["CGI"] = result["coverage_low_over_high"] / base_ratio

    summary = {
        "backbone": args.backbone, "k": args.k, "seed": args.seed,
        "training_shares": shares,
        "reference_images": len(reference), "baseline_images": len(baseline), "unseen_real_images": len(unseen),
        "probe_check": probe_check,
        "sets": results,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"probe balanced accuracy on held-out training images: {probe_check['balanced_accuracy_on_baseline']:.3f}")
    print(f"{'set':15s}{'HDAI':>7s}{'HDAI-on':>9s}{'TVD':>7s}{'off-man':>9s}{'cov H':>7s}{'cov L':>7s}{'CGI':>7s}"
          f"{'sim med':>9s}{'>0.95':>8s}")
    for name, r in results.items():
        g, cp = r["group_coverage"], r.get("copying", {})
        print(f"{name:15s}{r['probe_all']['HDAI']:7.3f}{r['probe_on_manifold']['HDAI']:9.3f}{r['probe_all']['TVD']:7.3f}"
              f"{r['off_manifold_fraction']:9.1%}{g['high']:7.3f}{g['low']:7.3f}{r['CGI']:7.3f}"
              f"{cp.get('dino_similarity_median', float('nan')):9.3f}"
              f"{cp.get(f'fraction_similarity_above_{COPY_SIMILARITY}', float('nan')):8.1%}")
        print(f"{'':15s}class shares: " + ", ".join(f"{c} {r['probe_all']['class_shares'][c]:.3f}" for c in CLASSES))


if __name__ == "__main__":
    main()
