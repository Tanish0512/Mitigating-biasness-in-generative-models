"""Build D1 from a random 75% sample of D0 and an unfiltered 25% G0 batch.

Generated images are classified exactly once and placed in ImageFolder class
directories. No confidence threshold or class-quota filtering is applied; the
unconditional G1 training script ignores these folder labels.

Example:
    python build_dataset1.py --g0-checkpoint g0.pt --r 0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm

from train import Diffusion, UNet
from train_resnet import build_model


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("dataset-0"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset-1"))
    parser.add_argument("--g0-checkpoint", type=Path, default=Path("g0.pt"))
    parser.add_argument("--classifier-checkpoint", type=Path, default=Path("models/checkpoint_best.pt"))
    parser.add_argument("--r", type=float, default=0.25, help="Fraction of D1 supplied by G0 samples")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as handle:
        if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"{path} is a Git LFS pointer, not a checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    return checkpoint


def main() -> None:
    args = parse_args()
    if not 0 < args.r < 1 or args.batch_size <= 0:
        raise ValueError("Require 0 < --r < 1 and positive --batch-size")
    manifest_path = args.output_dir / "manifest.csv"
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not manifest_path.is_file():
        raise FileExistsError(f"Non-empty output directory has no manifest: {args.output_dir}")
    for path in (args.g0_checkpoint, args.classifier_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = datasets.ImageFolder(args.source_dir)
    names = source.classes
    if not source.samples:
        raise ValueError(f"No images found under {args.source_dir}")
    if len(source) < 2:
        raise ValueError("Need at least two source images to create a mixed dataset")
    total_generated = round(len(source) * args.r)
    retained_count = len(source) - total_generated
    resume_partial = args.output_dir.exists() and manifest_path.is_file()
    retained_counts: Counter[str] = Counter()
    if resume_partial:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
        if len(existing_rows) != retained_count or any(row.get("source") != "dataset-0_sampled" for row in existing_rows):
            raise ValueError(
                f"Existing D1 manifest must contain exactly {retained_count} D0-sampled rows; "
                "refusing to overwrite or alter this dataset"
            )
        seen_sources: set[str] = set()
        for row in existing_rows:
            class_name = row.get("class")
            relative_source = row.get("source_path")
            if class_name not in names or not relative_source or relative_source in seen_sources:
                raise ValueError("Existing D1 manifest contains an invalid or duplicate source row")
            seen_sources.add(relative_source)
            if not (args.source_dir / relative_source).is_file():
                raise FileNotFoundError(f"D0 source listed in D1 manifest is missing: {relative_source}")
            if not (args.output_dir / class_name / row["filename"]).is_file():
                raise FileNotFoundError(f"D1 image listed in manifest is missing: {class_name}/{row['filename']}")
            retained_counts[class_name] += 1
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    g0 = load_checkpoint(args.g0_checkpoint)
    state = g0.get("model")
    if not isinstance(state, dict) or "class_embedding.weight" in state:
        raise ValueError("G0 checkpoint must contain an unconditional model")
    g0_args = g0.get("args", {})
    image_size = int(g0_args.get("image_size", 64))
    generator = UNet(num_classes=0, base_channels=int(g0_args.get("base_channels", 32))).to(device)
    generator.load_state_dict(g0.get("ema", state))
    generator.eval()
    diffusion = Diffusion(int(g0_args.get("timesteps", 1000)), device)

    classifier_ckpt = load_checkpoint(args.classifier_checkpoint)
    if classifier_ckpt.get("classes") != names:
        raise ValueError("Classifier class order does not match source dataset class order")
    classifier_size = int(classifier_ckpt.get("args", {}).get("image_size", 64))
    classifier = build_model(len(names), pretrained=False).to(device)
    classifier.load_state_dict(classifier_ckpt["model"])
    classifier.eval()
    preprocess = transforms.Compose([
        transforms.Resize((classifier_size, classifier_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (args.output_dir / name).mkdir(exist_ok=True)
    manifest = manifest_path.open("a" if resume_partial else "w", newline="", encoding="utf-8")
    writer = csv.writer(manifest)
    if not resume_partial:
        writer.writerow(["filename", "class", "source", "source_path", "classifier_confidence"])

    if not resume_partial:
        # Sample 75% from all of D0 without replacement.
        retained_indices = random.sample(range(len(source.samples)), retained_count)
        for index in tqdm(retained_indices, desc="sampling D0"):
            image_path, label = source.samples[index]
            class_name = names[label]
            filename = Path(image_path).name
            shutil.copy2(image_path, args.output_dir / class_name / filename)
            writer.writerow([filename, class_name, "dataset-0_sampled", str(Path(image_path).relative_to(args.source_dir)), ""])
            retained_counts[class_name] += 1

    generated_counts: Counter[str] = Counter()
    confidence_sums: Counter[str] = Counter()
    generated_index = 0
    with torch.inference_mode():
        for start in tqdm(range(0, total_generated, args.batch_size), desc="generating and classifying G0"):
            count = min(args.batch_size, total_generated - start)
            samples = diffusion.sample(generator, count, None, image_size).cpu()
            inputs = torch.stack([preprocess(transforms.ToPILImage()(sample)) for sample in samples]).to(device)
            probabilities = classifier(inputs).softmax(1).cpu()
            confidences, predictions = probabilities.max(1)
            for sample, prediction, confidence in zip(samples, predictions.tolist(), confidences.tolist()):
                class_name = names[prediction]
                filename = f"g0_sample_{generated_index:05d}.png"
                utils.save_image(sample, args.output_dir / class_name / filename)
                writer.writerow([filename, class_name, "G0_generated", f"sample_{generated_index:05d}", f"{confidence:.8f}"])
                generated_counts[class_name] += 1
                confidence_sums[class_name] += float(confidence)
                generated_index += 1
    manifest.close()

    final_counts = {name: retained_counts[name] + generated_counts[name] for name in names}
    summary = {
        "generation": "D1 for training G1",
        "source_dataset": str(args.source_dir),
        "g0_checkpoint": str(args.g0_checkpoint),
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "seed": args.seed,
        "replacement_fraction_r": args.r,
        "source_count": len(source),
        "retained_source_count": retained_count,
        "generated_count": total_generated,
        "total_count": retained_count + total_generated,
        "generated_selection": "All generated images retained after a single classifier pass; no confidence threshold or class balancing.",
        "class_order": names,
        "source_class_counts": {name: sum(label == i for _, label in source.samples) for i, name in enumerate(names)},
        "retained_source_class_counts": dict(retained_counts),
        "generated_class_counts_by_classifier": dict(generated_counts),
        "mean_classifier_confidence_by_generated_class": {
            name: confidence_sums[name] / generated_counts[name]
            for name in generated_counts
        },
        "final_class_folder_counts": final_counts,
        "classifier_preprocessing": {"resize": [classifier_size, classifier_size], "mean": MEAN, "std": STD},
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Built {args.output_dir}: {retained_count} D0 + {total_generated} unfiltered G0 = {len(source)} images")
    print(f"G0 fraction r={args.r:.1%}; generated class predictions={dict(generated_counts)}")


if __name__ == "__main__":
    main()
