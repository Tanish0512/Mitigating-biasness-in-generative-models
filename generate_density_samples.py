"""Generate an unlabeled image pool from the unconditional G0 DDPM.

The default is 3,800 images, matching the total of the project's proposed
1,000/200/50 class quotas. An unconditional model cannot be prompted to meet
those semantic class quotas; classify the generated pool separately if class
assignments are needed.

Example:
    python generate_density_samples.py \
        --checkpoint ddpm_g0/checkpoint_0050.pt \
        --output-dir generated-dataset-0 --batch-size 512
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torchvision import utils
from tqdm.auto import tqdm

from train import Diffusion, UNet


HIGH_DENSITY = {"aircraft", "bird", "car"}
MEDIUM_DENSITY = {"butterfly", "cat", "dog"}
LOW_DENSITY = {"fish", "flower", "human", "ship"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("ddpm_g0/checkpoint_0050.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated-dataset-0"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=3800)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but no Apple MPS device is available")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_samples <= 0:
        raise ValueError("--batch-size and --num-samples must be greater than zero")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    with args.checkpoint.open("rb") as handle:
        if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"{args.checkpoint} is a Git LFS pointer, not a checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{args.checkpoint} is not a training checkpoint with a 'model' state dict")

    state = checkpoint["model"]
    if "class_embedding.weight" in state:
        raise ValueError(
            "This checkpoint is class-conditional. Train an unconditional G0 checkpoint "
            "with train-g0.py before using this sampler."
        )

    saved_args = checkpoint.get("args", {})
    image_size = int(saved_args.get("image_size", 64))
    base_channels = int(saved_args.get("base_channels", 64))
    timesteps = int(saved_args.get("timesteps", 1000))
    if min(image_size, base_channels, timesteps) <= 0:
        raise ValueError("Checkpoint image_size, base_channels, and timesteps must be positive")

    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = UNet(num_classes=0, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint.get("ema", state))
    model.eval()
    diffusion = Diffusion(timesteps, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["sample_index", "filename"])
        with torch.inference_mode():
            for start in tqdm(range(0, args.num_samples, args.batch_size), desc="unconditional generation"):
                batch_count = min(args.batch_size, args.num_samples - start)
                samples = diffusion.sample(model, batch_count, None, image_size).cpu()
                for offset, sample in enumerate(samples):
                    index = start + offset
                    filename = f"sample_{index:05d}.png"
                    utils.save_image(sample, args.output_dir / filename)
                    writer.writerow([index, filename])

    summary = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "image_size": image_size,
        "total_images": args.num_samples,
        "class_counts": None,
        "note": "Unconditional samples have no requested class labels; classify separately to measure class distribution.",
        "previous_class_quota_targets": {
            "high": {name: 1000 for name in sorted(HIGH_DENSITY)},
            "medium": {name: 200 for name in sorted(MEDIUM_DENSITY)},
            "low": {name: 50 for name in sorted(LOW_DENSITY)},
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {summary['total_images']} unlabeled images in {args.output_dir}")


if __name__ == "__main__":
    main()
