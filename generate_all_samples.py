"""Generate 10 samples per class from every checkpoint in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets, utils
from tqdm.auto import tqdm

from train import Diffusion, UNet


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as handle:
        if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"{path} is a Git LFS pointer, not a checkpoint")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"{path} is not a valid training checkpoint")
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("ddpm_ema_attn_large"))
    parser.add_argument("--data-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0 or args.batch_size <= 0:
        raise ValueError("Sample counts and batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    class_names = datasets.ImageFolder(args.data_dir).classes
    checkpoints = sorted(args.checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_*.pt files found in {args.checkpoint_dir}")

    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(checkpoint_path)
        state = checkpoint["model"]
        class_embedding = state.get("class_embedding.weight")
        if class_embedding is None:
            raise ValueError(f"{checkpoint_path} is unconditional")
        num_classes = class_embedding.shape[0]
        if len(class_names) != num_classes:
            raise ValueError(f"Class count mismatch for {checkpoint_path}")

        saved_args = checkpoint.get("args", {})
        image_size = int(saved_args.get("image_size", 64))
        base_channels = int(saved_args.get("base_channels", 64))
        timesteps = int(saved_args.get("timesteps", 1000))
        model = UNet(num_classes=num_classes, base_channels=base_channels).to(device)
        model.load_state_dict(checkpoint.get("ema", state))
        model.eval()
        diffusion = Diffusion(timesteps, device)

        model_number = checkpoint_path.stem.removeprefix("checkpoint_")
        model_dir = args.output_dir / f"model_{model_number}"
        labels = [label for label in range(num_classes) for _ in range(args.samples_per_class)]
        samples = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(labels), args.batch_size),
                              desc=checkpoint_path.stem):
                batch_labels = torch.tensor(labels[start:start + args.batch_size],
                                            dtype=torch.long, device=device)
                samples.append(diffusion.sample(model, len(batch_labels), batch_labels, image_size).cpu())
        samples = torch.cat(samples)

        offset = 0
        for label, class_name in enumerate(class_names):
            class_dir = model_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            class_samples = samples[offset:offset + args.samples_per_class]
            for index, image in enumerate(class_samples):
                utils.save_image(image, class_dir / f"sample_{index:02d}.png")
            offset += args.samples_per_class

        print(f"Saved {len(labels)} samples to {model_dir}")


if __name__ == "__main__":
    main()
