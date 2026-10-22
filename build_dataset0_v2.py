"""Build dataset-0-v2: six classes, high/low density only, 20% synthetic.

High-density classes (aircraft, car, ship) get 6,000 images each and
low-density classes (butterfly, fish, human) 300 each, a 20:1 ratio. Each class
is 80% real and 20% synthetic. The source "fish" class (sharks, whales and
dolphins) is written out as ``marine_animal``.

Sources, in order of preference:

* real: old D0 real images, then unused images from the source pool
  (``images/``), skipping exact duplicates by SHA-256;
* synthetic: old D0 synthetic images (large DDPM, ``sample_*``), then new
  images from generate_synthetic.py.

Screening rule, applied to every image: reject it when both classifiers
disagree with its class folder. The classifiers are a class-balanced logistic
probe on DINOv2 features, trained on all real old-D0 images (10 classes), and
the project ResNet-18. Old-D0 real images get 5-fold out-of-fold probe
predictions, so no image is judged by a probe that saw it. ``--path-filter``
keeps only pool images whose original source path matches a regular
expression, for example to restrict fish to sharks.

Example::

    python build_dataset0_v2.py --pool-dir images --d0-dir dataset-0 \\
        --synthetic-dirs synthetic_v2/a synthetic_v2/b synthetic_v2/c \\
        --resnet-checkpoint models/checkpoint_best.pt --output-dir dataset-0-v2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from torchvision import transforms
from tqdm.auto import tqdm

from train_resnet import build_model


# Alphabetical ImageFolder order of the original ten-class data; the ResNet uses it.
CLASSES = ["aircraft", "bird", "butterfly", "car", "cat", "dog", "fish", "flower", "human", "ship"]
GROUPS = {"high": ["aircraft", "car", "ship"], "low": ["butterfly", "fish", "human"]}
# The pool's "fish" folder is sharks, whales and dolphins, so v2 names the class honestly.
OUTPUT_NAMES = {"fish": "marine_animal"}
COUNTS = {"high": 6000, "low": 300}
BACKBONE = "facebook/dinov2-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-dir", type=Path, required=True, help="Ten-class source pool (images/)")
    parser.add_argument("--d0-dir", type=Path, required=True, help="Old ten-class dataset-0")
    parser.add_argument("--synthetic-dirs", type=Path, nargs="+", required=True,
                        help="Output folders of generate_synthetic.py")
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("feature_cache"), help="DINOv2 feature cache")
    parser.add_argument("--path-filter", action="append", default=[], metavar="CLASS=REGEX",
                        help="Keep only pool images whose original_path matches REGEX")
    parser.add_argument("--synthetic-fraction", type=float, default=0.2)
    parser.add_argument("--screen-margin", type=float, default=1.3, help="Pool candidates screened per needed image")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pool_origins(pool_dir: Path, cls: str) -> dict[str, str]:
    manifest = pool_dir / cls / "manifest.csv"
    if not manifest.is_file():
        return {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: row["original_path"] for row in csv.DictReader(handle)}


class Features:
    """L2-normalised DINOv2 CLS features, cached per named image list."""

    def __init__(self, device: torch.device, batch_size: int, cache_dir: Path):
        from transformers import AutoImageProcessor, AutoModel

        self.model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(BACKBONE)
        self.device, self.batch_size, self.cache_dir = device, batch_size, cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def __call__(self, name: str, paths: list[Path]) -> np.ndarray:
        cache = self.cache_dir / f"{name}.pt"
        keys = [str(p) for p in paths]
        if cache.is_file():
            stored = torch.load(cache, weights_only=True)
            if stored["paths"] == keys:
                return stored["features"].numpy()
        chunks = []
        for start in tqdm(range(0, len(paths), self.batch_size), desc=f"features {name}"):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + self.batch_size]]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            cls = self.model(**inputs).last_hidden_state[:, 0]
            chunks.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu())
        features = torch.cat(chunks)
        torch.save({"paths": keys, "features": features}, cache)
        return features.numpy()


class ResNet:
    def __init__(self, path: Path, device: torch.device, batch_size: int):
        # Safe loading; Linux-saved PosixPath arguments become PurePosixPath on any OS.
        torch.serialization.add_safe_globals([(PurePosixPath, "pathlib.PosixPath")])
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if list(checkpoint.get("classes", ())) != CLASSES:
            raise ValueError(f"ResNet class order {checkpoint.get('classes')} does not match {CLASSES}")
        size = int(checkpoint.get("args", {}).get("image_size", 64))
        self.model = build_model(len(CLASSES), pretrained=False)
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(device).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.device, self.batch_size = device, batch_size

    @torch.inference_mode()
    def __call__(self, paths: list[Path]) -> list[str]:
        labels = []
        for start in tqdm(range(0, len(paths), self.batch_size), desc="resnet"):
            batch = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + self.batch_size]])
            labels += [CLASSES[i] for i in self.model(batch.to(self.device)).argmax(1).tolist()]
        return labels


class Screen:
    """Reject an image when both the DINOv2 probe and the ResNet disagree with its class."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.features = Features(device, args.batch_size, args.cache_dir)
        self.resnet = ResNet(args.resnet_checkpoint, device, args.batch_size)
        d0 = [(p, c) for c in CLASSES for p in sorted((args.d0_dir / c).glob("*.png"))]
        paths = [p for p, _ in d0]
        labels = np.array([CLASSES.index(c) for _, c in d0])
        real = np.array([not p.name.startswith("sample_") for p in paths])
        features = self.features("d0", paths)
        self.probe = LogisticRegression(max_iter=3000, class_weight="balanced")
        predicted = np.empty(len(d0), dtype=int)
        folds = StratifiedKFold(5, shuffle=True, random_state=args.seed)
        predicted[real] = cross_val_predict(self.probe, features[real], labels[real], cv=folds)
        self.probe.fit(features[real], labels[real])
        predicted[~real] = self.probe.predict(features[~real])
        resnet = self.resnet(paths)
        self.old_d0_ok = {
            p.resolve(): CLASSES[i] == c or r == c for (p, c), i, r in zip(d0, predicted, resnet)
        }
        self.calls = 0

    def __call__(self, paths: list[Path], cls: str) -> list[bool]:
        if not paths:
            return []
        self.calls += 1
        probe = [CLASSES[i] for i in self.probe.predict(self.features(f"screen_{cls}_{self.calls}", paths))]
        return [p == cls or r == cls for p, r in zip(probe, self.resnet(paths))]


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    filters = dict(item.split("=", 1) for item in args.path_filter)
    screen = Screen(args, device)

    rows, report, seen = [], {}, set()
    for group, members in GROUPS.items():
        for cls in members:
            total = COUNTS[group]
            need_synthetic = round(total * args.synthetic_fraction)
            need_real = total - need_synthetic
            origins = pool_origins(args.pool_dir, cls)
            pattern = re.compile(filters[cls]) if cls in filters else None
            allowed = lambda name: pattern is None or bool(pattern.search(origins.get(name, "")))
            stats = Counter()
            old_files = sorted((args.d0_dir / cls).glob("*.png"))
            stats["old_d0_rejected"] = sum(not screen.old_d0_ok[p.resolve()] for p in old_files)

            # Real: screened old-D0 real images first (subject to the same path filter), then the pool.
            old_real = [p for p in old_files
                        if not p.name.startswith("sample_") and screen.old_d0_ok[p.resolve()] and allowed(p.name)]
            rng.shuffle(old_real)
            chosen_real = []
            for path in old_real:
                digest = sha256(path)
                if digest not in seen and len(chosen_real) < need_real:
                    seen.add(digest)
                    chosen_real.append((path, "real_d0", digest))
            old_names = {p.name for p in old_files}
            candidates = [p for p in sorted((args.pool_dir / cls).glob("*.png")) if p.name not in old_names and allowed(p.name)]
            rng.shuffle(candidates)
            while len(chosen_real) < need_real and candidates:
                take = max(1, int((need_real - len(chosen_real)) * args.screen_margin))
                batch, candidates = candidates[:take], candidates[take:]
                for path, ok in zip(batch, screen(batch, cls)):
                    digest = sha256(path)
                    if not ok:
                        stats["pool_rejected"] += 1
                    elif digest in seen:
                        stats["pool_duplicate"] += 1
                    elif len(chosen_real) < need_real:
                        seen.add(digest)
                        chosen_real.append((path, "real_pool", digest))
            if len(chosen_real) < need_real:
                raise RuntimeError(f"{cls}: only {len(chosen_real)} of {need_real} real images available")

            # Synthetic: screened old-D0 synthetic first, then screened new images.
            old_synthetic = [p for p in old_files if p.name.startswith("sample_") and screen.old_d0_ok[p.resolve()]]
            rng.shuffle(old_synthetic)
            chosen_synthetic = [(p, "synthetic_d0", sha256(p)) for p in old_synthetic[:need_synthetic]]
            if len(chosen_synthetic) < need_synthetic:
                new = [p for d in args.synthetic_dirs for p in sorted((d / cls).glob("*.png"))]
                rng.shuffle(new)
                for path, ok in zip(new, screen(new, cls)):
                    if not ok:
                        stats["synthetic_rejected"] += 1
                    elif len(chosen_synthetic) < need_synthetic:
                        chosen_synthetic.append((path, "synthetic_v2", sha256(path)))
            if len(chosen_synthetic) < need_synthetic:
                raise RuntimeError(f"{cls}: only {len(chosen_synthetic)} of {need_synthetic} synthetic images available")

            name = OUTPUT_NAMES.get(cls, cls)
            folder = args.output_dir / name
            folder.mkdir(parents=True, exist_ok=True)
            for path, source, digest in chosen_real + chosen_synthetic:
                shutil.copy2(path, folder / path.name)
                rows.append([path.name, name, group, source, str(path), origins.get(path.name, ""), digest])
            report[name] = {"group": group, "total": total, **Counter(s for _, s, _ in chosen_real + chosen_synthetic), **stats}
            print(name, report[name])

    names = Counter((r[1], r[0]) for r in rows)
    if max(names.values()) > 1:
        raise RuntimeError("Filename collision inside a class folder")
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "class", "density_group", "source", "copied_from", "pool_original_path", "sha256"])
        writer.writerows(rows)
    summary = {
        "classes": [OUTPUT_NAMES.get(c, c) for members in GROUPS.values() for c in members],
        "renamed_classes": {OUTPUT_NAMES[c]: f"source class '{c}' (sharks, whales, dolphins)" for c in OUTPUT_NAMES},
        "density_groups": {g: [OUTPUT_NAMES.get(c, c) for c in m] for g, m in GROUPS.items()},
        "per_class_counts": COUNTS,
        "synthetic_fraction": args.synthetic_fraction,
        "total_images": len(rows),
        "seed": args.seed,
        "path_filters": filters,
        "screening": "rejected when both the DINOv2 probe (trained on real old-D0) and the ResNet disagree with the class",
        "per_class": report,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Built {args.output_dir}: {len(rows)} images")


if __name__ == "__main__":
    main()
