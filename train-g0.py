"""Train a compact generation-zero DDPM on dataset-0.

    python train-g0.py --epochs 50

Uses an unconditional attention U-Net with half the default channel width and
EMA weights for previews. The class folders provide images for training, but
their labels are not passed to the model.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    """Standard DDPM sinusoidal embedding of the diffusion timestep."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / max(half, 1)
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
        self.time = nn.Linear(time_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(t).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Self-attention at low resolution with timestep/class context."""
    def __init__(self, channels: int, time_dim: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.condition = nn.Linear(time_dim, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.condition(t).unsqueeze(1), tokens), dim=1)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        attended = attended[:, 1:].transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.projection(attended)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, attention: bool = False):
        super().__init__()
        self.block1 = Block(in_channels, out_channels, time_dim)
        self.block2 = Block(out_channels, out_channels, time_dim)
        self.attention = SelfAttention(out_channels, time_dim) if attention else None
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block2(self.block1(x, t), t)
        if self.attention is not None:
            h = self.attention(h, t)
        return h, self.down(h)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int, attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block1 = Block(out_channels + skip_channels, out_channels, time_dim)
        self.block2 = Block(out_channels, out_channels, time_dim)
        self.attention = SelfAttention(out_channels, time_dim) if attention else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = self.block1(torch.cat((x, skip), dim=1), t)
        x = self.block2(x, t)
        return self.attention(x, t) if self.attention is not None else x


class UNet(nn.Module):
    def __init__(self, num_classes: int = 0, base_channels: int = 32, time_dim: int = 256):
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm")
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.class_embedding = nn.Embedding(num_classes, time_dim) if num_classes else None
        self.input = nn.Conv2d(3, base_channels, 3, padding=1)
        self.down1 = Down(base_channels, base_channels * 2, time_dim)      # 64 -> 32
        self.down2 = Down(base_channels * 2, base_channels * 4, time_dim)  # 32 -> 16
        self.down3 = Down(base_channels * 4, base_channels * 8, time_dim, attention=True)  # attention at 16x16
        self.mid1 = Block(base_channels * 8, base_channels * 8, time_dim)
        self.mid_attention = SelfAttention(base_channels * 8, time_dim)  # attention at 8x8
        self.mid2 = Block(base_channels * 8, base_channels * 8, time_dim)
        self.up3 = Up(base_channels * 8, base_channels * 8, base_channels * 4, time_dim, attention=True)
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
        skip3, x = self.down3(x, t)
        x = self.mid_attention(self.mid1(x, t), t)
        x = self.mid2(x, t)
        x = self.up3(x, skip3, t)
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


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def copy_from(self, model: nn.Module) -> None:
        self.shadow = {name: value.detach().clone() for name, value in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: value.clone() for name, value in state.items()}


def save_checkpoint(path: Path, model: nn.Module, ema: EMA, optimizer: torch.optim.Optimizer, epoch: int, args: argparse.Namespace, scaler: torch.amp.GradScaler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "args": vars(args), "classes": args.classes, "scaler": scaler.state_dict()}, path)


def save_loss_history(output_dir: Path, epoch: int, loss: float, resume: bool) -> None:
    """Append the epoch loss to CSV and regenerate the loss plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "loss.csv"
    if not csv_path.exists() or (not resume and epoch == 1):
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
    parser.add_argument("--data-dir", type=Path, default=Path("dataset-0"))
    parser.add_argument("--output-dir", type=Path, default=Path("ddpm_g0"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    for name in ("batch_size", "epochs", "timesteps", "save_every", "sample_count"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.image_size < 8 or args.image_size % 8:
        parser.error("--image-size must be at least 8 and divisible by 8")
    if args.base_channels < 8 or args.base_channels % 8:
        parser.error("--base-channels must be a positive multiple of 8")
    if args.num_workers < 0 or args.learning_rate <= 0 or not 0 <= args.ema_decay < 1:
        parser.error("Require num-workers >= 0, learning-rate > 0 and 0 <= ema-decay < 1")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(args.device)
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
    args.classes = dataset.classes
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0, drop_last=False)
    model = UNet(0, args.base_channels).to(device)
    ema = EMA(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    diffusion = Diffusion(args.timesteps, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if any(name.startswith("class_embedding.") for name in checkpoint["model"]):
            raise ValueError("Cannot resume a class-conditional checkpoint as unconditional G0; start a new run")
        if checkpoint.get("classes", dataset.classes) != dataset.classes:
            raise ValueError("Checkpoint class ordering does not match the dataset")
        model.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        else:
            ema.copy_from(model)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate

        start_epoch = checkpoint["epoch"] + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(dataset.targets)
    (args.output_dir / "dataset.json").write_text(json.dumps({
        "data_dir": str(args.data_dir), "classes": dataset.classes,
        "class_counts": {name: counts[idx] for idx, name in enumerate(dataset.classes)},
        "total_images": len(dataset),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}; base channels={args.base_channels}; EMA={args.ema_decay}")
    print(f"Training unconditionally on {len(dataset):,} images from folders={dataset.classes}, device={device}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for images, _folder_labels in progress:
            images = images.to(device, non_blocking=True)
            # ImageFolder supplies labels for bookkeeping; G0 intentionally ignores them.
            timesteps = torch.randint(0, args.timesteps, (images.size(0),), device=device)
            noisy_images, noise = diffusion.noise_images(images, timesteps)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.mse_loss(model(noisy_images, timesteps, None), noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                ema.update(model)
            running_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = running_loss / len(dataset)
        print(f"epoch {epoch + 1}: mean loss={mean_loss:.5f}")
        save_loss_history(args.output_dir, epoch + 1, mean_loss, args.resume is not None)
        model.eval()
        # Use EMA weights for one unconditional sample grid after every epoch.
        # Restore the training weights afterward so the next epoch continues correctly.
        training_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        try:
            ema.copy_to(model)
            samples = diffusion.sample(model, args.sample_count, None, args.image_size)
            utils.save_image(samples, args.output_dir / f"samples_{epoch + 1:04d}.png",
                             nrow=max(1, int(math.sqrt(args.sample_count))))
        finally:
            model.load_state_dict(training_state)
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(args.output_dir / f"checkpoint_{epoch + 1:04d}.pt", model, ema, optimizer, epoch, args, scaler)


if __name__ == "__main__":
    main()
