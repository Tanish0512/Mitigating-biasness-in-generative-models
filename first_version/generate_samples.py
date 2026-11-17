"""Generate classwise samples from a trained DDPM checkpoint.

Class order is the alphabetical ImageFolder order:
    aircraft, bird, butterfly, car, cat, dog, fish, flower, human, ship

Examples:
    # Generate 10 images from every class
    python generate_samples.py --checkpoint ddpm_runs/checkpoint_0050.pt --n 10

    # Generate 10 fish, 20 flowers, and 30 humans (all other classes: 0)
    python generate_samples.py --checkpoint ddpm_runs/checkpoint_0050.pt \
        --counts 0 0 10 0 0 0 20 0 30 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets, utils
from tqdm.auto import tqdm

from train_ddpm import Diffusion, UNet


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint and provide a useful error for missing Git LFS files."""
    with path.open("rb") as handle:
        header = handle.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the actual checkpoint. "
            "Install Git LFS and run `git lfs pull`, or copy the binary checkpoint "
            "into this path."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of samples per class when --counts is not supplied")
    parser.add_argument("--counts", type=int, nargs="+", default=None, metavar="COUNT",
                        help="One count per ImageFolder class, in class order; overrides --n")
    parser.add_argument("--data-dir", type=Path, default=Path("images"),
                        help="Dataset root used to recover class names and ordering")
    parser.add_argument("--output-dir", type=Path, default=Path("generated_samples"))
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Sampling batch size to limit GPU memory use")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 0 or (args.counts is not None and any(count < 0 for count in args.counts)):
        raise ValueError("Sample counts must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")

    checkpoint = load_checkpoint(args.checkpoint)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{args.checkpoint} is not a training checkpoint with a 'model' state dict")
    model_state = checkpoint["model"]
    class_embedding = model_state.get("class_embedding.weight")
    if class_embedding is None:
        raise ValueError(
            "The checkpoint is unconditional. generate_samples.py requires a class-conditional checkpoint."
        )
    checkpoint_num_classes = class_embedding.shape[0]
    saved_args = checkpoint.get("args", {})
    image_size = int(saved_args.get("image_size", 64))
    base_channels = int(saved_args.get("base_channels", 64))
    timesteps = int(saved_args.get("timesteps", 1000))
    if image_size <= 0 or timesteps <= 0:
        raise ValueError("The checkpoint must contain positive image_size and timesteps values")

    dataset = datasets.ImageFolder(args.data_dir)
    class_names = dataset.classes
    if len(class_names) != checkpoint_num_classes:
        raise ValueError(
            "The dataset/checkpoint class counts do not match: "
            f"dataset has {len(class_names)} classes ({class_names}), "
            f"checkpoint has {checkpoint_num_classes}"
        )
    counts = args.counts if args.counts is not None else [args.n] * len(class_names)
    if len(counts) != len(class_names):
        raise ValueError(f"Expected one count for each of these classes: {class_names}")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    model = UNet(num_classes=checkpoint_num_classes, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint.get("ema", model_state))
    model.eval()
    diffusion = Diffusion(timesteps, device)

    requested_labels = [
        label for label, count in enumerate(counts) for _ in range(count)
    ]
    if not requested_labels:
        print("No samples requested.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(requested_labels), args.batch_size), desc="generating"):
            labels = torch.tensor(
                requested_labels[start:start + args.batch_size], dtype=torch.long, device=device
            )
            generated.append(diffusion.sample(model, len(labels), labels, image_size).cpu())
    samples = torch.cat(generated)

    offset = 0
    for label, (class_name, count) in enumerate(zip(class_names, counts)):
        class_dir = args.output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_samples = samples[offset:offset + count]
        for index, image in enumerate(class_samples):
            utils.save_image(image, class_dir / f"sample_{index:05d}.png")
        if count:
            utils.save_image(class_samples, args.output_dir / f"{class_name}_grid.png",
                             nrow=min(8, count))
        offset += count
    print(f"Generated {len(requested_labels)} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
