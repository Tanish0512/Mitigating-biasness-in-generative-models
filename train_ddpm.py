"""Train a small DDPM on the image folders in this repository.

Expected dataset layout::

    images/
      cat/*.png
      dog/*.png
      ...

Example:
    python train_ddpm.py --data-dir images --epochs 50 --batch-size 64

The model is class-conditional when multiple class folders are present.  Use
``--unconditional`` to train one shared distribution instead.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm.auto import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    values = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((values.sin(), values.cos()), dim=1)
    return F.pad(embedding, (0, dim % 2))


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.block = Block(in_channels, out_channels, time_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x, t)
        return h, self.down(h)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = Block(out_channels + skip_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.block(torch.cat((x, skip), dim=1), t)


class UNet(nn.Module):
    def __init__(self, num_classes: int = 0, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.class_embedding = nn.Embedding(num_classes, time_dim) if num_classes else None
        self.input = nn.Conv2d(3, base_channels, 3, padding=1)
        self.down1 = Down(base_channels, base_channels * 2, time_dim)      # 64 -> 32
        self.down2 = Down(base_channels * 2, base_channels * 4, time_dim)  # 32 -> 16
        self.mid1 = Block(base_channels * 4, base_channels * 8, time_dim)
        self.mid2 = Block(base_channels * 8, base_channels * 4, time_dim)
        self.up2 = Up(base_channels * 4, base_channels * 4, base_channels * 2, time_dim)
        self.up1 = Up(base_channels * 2, base_channels * 2, base_channels, time_dim)
        self.output = nn.Sequential(nn.GroupNorm(8, base_channels), nn.SiLU(), nn.Conv2d(base_channels, 3, 3, padding=1))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        t = self.time_mlp(sinusoidal_embedding(timesteps, self.time_mlp[0].in_features))
        if self.class_embedding is not None:
            if labels is None:
                raise ValueError("Labels are required for a class-conditional model")
            t = t + self.class_embedding(labels)
        x = self.input(x)
        skip1, x = self.down1(x, t)
        skip2, x = self.down2(x, t)
        x = self.mid2(self.mid1(x, t), t)
        x = self.up2(x, skip2, t)
        x = self.up1(x, skip1, t)
        return self.output(x)


class Diffusion:
    def __init__(self, steps: int, device: torch.device):
        self.steps = steps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        return alpha_bar.sqrt() * x + (1 - alpha_bar).sqrt() * noise, noise

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, labels: torch.Tensor | None, image_size: int) -> torch.Tensor:
        x = torch.randn(n, 3, image_size, image_size, device=self.device)
        for step in tqdm(range(self.steps - 1, -1, -1), desc="sampling", leave=False):
            t = torch.full((n,), step, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t, labels)
            alpha = self.alpha[t][:, None, None, None]
            alpha_bar = self.alpha_bar[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            mean = (x - beta * predicted_noise / (1 - alpha_bar).sqrt()) / alpha.sqrt()
            if step > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return ((x.clamp(-1, 1) + 1) / 2).clamp(0, 1)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "args": vars(args)}, path)


def save_loss_history(output_dir: Path, epoch: int, loss: float, resume: bool) -> None:
    """Append the epoch loss to CSV and regenerate the loss plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "loss.csv"
    if not resume and epoch == 1:
        csv_path.write_text("epoch,mean_loss\n", encoding="utf-8")
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([epoch, f"{loss:.8f}"])

    epochs, losses = [], []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["mean_loss"]))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o", linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Mean training loss")
    plt.title("DDPM training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("ddpm_runs"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--unconditional", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if not dataset.samples:
        raise RuntimeError(f"No images found in {args.data_dir}")
    conditional = not args.unconditional and len(dataset.classes) > 1
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0, drop_last=True)
    model = UNet(0 if not conditional else len(dataset.classes), args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    diffusion = Diffusion(args.timesteps, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    print(f"Training on {len(dataset):,} images, classes={dataset.classes}, device={device}, conditional={conditional}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) if conditional else None
            timesteps = torch.randint(0, args.timesteps, (images.size(0),), device=device)
            noisy_images, noise = diffusion.noise_images(images, timesteps)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.mse_loss(model(noisy_images, timesteps, labels), noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / max(len(loader), 1)
        print(f"epoch {epoch + 1}: mean loss={mean_loss:.5f}")
        save_loss_history(args.output_dir, epoch + 1, mean_loss, args.resume is not None)
        model.eval()
        sample_labels = None
        if conditional:
            sample_labels = torch.arange(args.sample_count, device=device) % len(dataset.classes)
        samples = diffusion.sample(model, args.sample_count, sample_labels, args.image_size)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        utils.save_image(samples, args.output_dir / f"samples_{epoch + 1:04d}.png", nrow=int(math.sqrt(args.sample_count)))
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(args.output_dir / f"checkpoint_{epoch + 1:04d}.pt", model, optimizer, epoch, args)


if __name__ == "__main__":
    main()
