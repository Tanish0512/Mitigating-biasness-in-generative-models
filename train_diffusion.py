"""Train an unconditional DDPM from scratch; one recipe for G0, G1 and mitigated runs.

Every run uses the same settings, so differences between generations come from
the training data alone:

* U-Net with attention at 16x16 and 8x8, base width 48 (22.4M parameters),
  dropout 0.1;
* cosine noise schedule (Nichol & Dhariwal, 2021), 1,000 steps, epsilon
  prediction with MSE loss;
* 150,000 optimizer steps, global batch 128, AdamW at 2e-4 with 5,000 linear
  warmup steps then constant, gradient clipping at 1.0, EMA 0.9999;
* bf16 autocast, horizontal flips only;
* the step-150,000 EMA weights are the model of record (``final.pt``).

Class folders only define the image mixture; labels are never given to the
model. The whole dataset is loaded onto each GPU as uint8. Multi-GPU runs use
DistributedDataParallel and split the global batch across GPUs.

Examples::

    python train_diffusion.py --data-dir dataset-0-v2 --output-dir runs/g0
    torchrun --nproc_per_node 3 train_diffusion.py --data-dir dataset-0-v2 --output-dir runs/g0

Re-running the same command resumes from the newest checkpoint in --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torchvision import utils
from tqdm.auto import tqdm


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / max(half, 1))
    values = timesteps.float()[:, None] * frequencies[None, :]
    return F.pad(torch.cat((values.sin(), values.cos()), dim=1), (0, dim % 2))


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels * 2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(t).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
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
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float, attention: bool = False):
        super().__init__()
        self.block1 = Block(in_channels, out_channels, time_dim, dropout)
        self.block2 = Block(out_channels, out_channels, time_dim, dropout)
        self.attention = SelfAttention(out_channels, time_dim) if attention else None
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block2(self.block1(x, t), t)
        if self.attention is not None:
            h = self.attention(h, t)
        return h, self.down(h)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int, dropout: float,
                 attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block1 = Block(out_channels + skip_channels, out_channels, time_dim, dropout)
        self.block2 = Block(out_channels, out_channels, time_dim, dropout)
        self.attention = SelfAttention(out_channels, time_dim) if attention else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.block2(self.block1(torch.cat((self.up(x), skip), dim=1), t), t)
        return self.attention(x, t) if self.attention is not None else x


class UNet(nn.Module):
    """Unconditional U-Net: 64 -> 32 -> 16 -> 8, attention at 16x16 and 8x8."""

    def __init__(self, base_channels: int = 48, time_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        if base_channels % 8:
            raise ValueError("base_channels must be divisible by 8 for GroupNorm")
        c = base_channels
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(3, c, 3, padding=1)
        self.down1 = Down(c, c * 2, time_dim, dropout)
        self.down2 = Down(c * 2, c * 4, time_dim, dropout)
        self.down3 = Down(c * 4, c * 8, time_dim, dropout, attention=True)
        self.mid1 = Block(c * 8, c * 8, time_dim, dropout)
        self.mid_attention = SelfAttention(c * 8, time_dim)
        self.mid2 = Block(c * 8, c * 8, time_dim, dropout)
        self.up3 = Up(c * 8, c * 8, c * 4, time_dim, dropout, attention=True)
        self.up2 = Up(c * 4, c * 4, c * 2, time_dim, dropout)
        self.up1 = Up(c * 2, c * 2, c, time_dim, dropout)
        self.output = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(sinusoidal_embedding(timesteps, self.time_mlp[0].in_features))
        x = self.input(x)
        skip1, x = self.down1(x, t)
        skip2, x = self.down2(x, t)
        skip3, x = self.down3(x, t)
        x = self.mid2(self.mid_attention(self.mid1(x, t), t), t)
        x = self.up1(self.up2(self.up3(x, skip3, t), skip2, t), skip1, t)
        return self.output(x)


class Diffusion:
    """Cosine-schedule DDPM with ancestral sampling and clipped x0 predictions."""

    def __init__(self, steps: int, device: torch.device, schedule: str = "cosine"):
        self.steps = steps
        if schedule == "cosine":
            s = 0.008
            x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
            alpha_bar = torch.cos((x / steps + s) / (1 + s) * math.pi / 2) ** 2
            beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
        elif schedule == "linear":
            beta = torch.linspace(1e-4, 0.02, steps, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule {schedule}")
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, 0)
        alpha_bar_prev = torch.cat((torch.ones(1, dtype=torch.float64), alpha_bar[:-1]))
        to = lambda v: v.float().to(device)
        self.beta, self.alpha_bar = to(beta), to(alpha_bar)
        self.posterior_variance = to(beta * (1 - alpha_bar_prev) / (1 - alpha_bar))
        self.mean_x0 = to(beta * alpha_bar_prev.sqrt() / (1 - alpha_bar))
        self.mean_xt = to((1 - alpha_bar_prev) * alpha.sqrt() / (1 - alpha_bar))

    def noise_images(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        a = self.alpha_bar[t][:, None, None, None]
        return a.sqrt() * x + (1 - a).sqrt() * noise, noise

    @torch.no_grad()
    def sample(self, model: nn.Module, n: int, image_size: int, generator: torch.Generator | None = None,
               progress: bool = False) -> torch.Tensor:
        device = self.beta.device
        x = torch.randn(n, 3, image_size, image_size, device=device, generator=generator)
        for step in tqdm(range(self.steps - 1, -1, -1), desc="sampling", leave=False, disable=not progress):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                eps = model(x, t).float()
            a = self.alpha_bar[step]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            x = self.mean_x0[step] * x0 + self.mean_xt[step] * x
            if step > 0:
                noise = torch.randn(x.shape, device=device, generator=generator)
                x = x + self.posterior_variance[step].sqrt() * noise
        return (x.clamp(-1, 1) + 1) / 2


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch-size", type=int, default=128, help="Global batch size across all GPUs")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-steps-this-run", type=int, default=None, help="Stop early (benchmarking)")
    return parser.parse_args()


def load_images(data_dir: Path, image_size: int) -> tuple[torch.Tensor, dict[str, int]]:
    """All images of all class folders as one uint8 NCHW tensor; class counts for the record."""
    paths, counts = [], {}
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
        counts[folder.name] = len(files)
        paths += files
    if not paths:
        raise RuntimeError(f"No images found in {data_dir}")
    array = np.stack([np.asarray(Image.open(p).convert("RGB").resize((image_size, image_size), Image.BICUBIC))
                      for p in tqdm(paths, desc="loading images", leave=False)])
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous(), counts


def main() -> None:
    args = parse_args()
    distributed = "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    if args.batch_size % world:
        raise ValueError(f"--batch-size {args.batch_size} must divide evenly across {world} GPUs")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    main_process = rank == 0
    per_gpu = args.batch_size // world

    images, counts = load_images(args.data_dir, args.image_size)
    images = images.to(device)
    model = UNet(args.base_channels, dropout=args.dropout).to(device)
    ema = EMA(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    diffusion = Diffusion(args.timesteps, device, args.schedule)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    checkpoints = sorted(args.output_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        state = torch.load(checkpoints[-1], map_location=device, weights_only=False)
        if state["args"]["data_dir"] != str(args.data_dir):
            raise ValueError(f"Resume data dir {state['args']['data_dir']} != {args.data_dir}")
        model.load_state_dict(state["model"])
        ema.shadow = state["ema"]
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        if main_process:
            print(f"Resumed from {checkpoints[-1]} at step {step}")
    net = DistributedDataParallel(model, device_ids=[local_rank]) if distributed else model

    if main_process:
        record = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        record.update(world_size=world, per_gpu_batch=per_gpu, class_counts=counts, total_images=len(images),
                      parameters=sum(p.numel() for p in model.parameters()))
        (args.output_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"{record['parameters']:,} parameters; {len(images):,} images {counts}; {world} GPU(s) x {per_gpu}")
        log_path = args.output_dir / "loss.csv"
        if not log_path.exists():
            log_path.write_text("step,loss,learning_rate,seconds\n", encoding="utf-8")

    def save(path: Path) -> None:
        torch.save({"model": model.state_dict(), "ema": ema.shadow, "optimizer": optimizer.state_dict(), "step": step,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "class_counts": counts}, path)

    # Same permutation on every rank; each rank takes its own slice of every global batch.
    # An epoch is every full global batch of one permutation; a partial last batch is dropped.
    epoch_rng = torch.Generator().manual_seed(args.seed)
    steps_per_epoch = len(images) // args.batch_size
    order = torch.randperm(len(images), generator=epoch_rng)
    for _ in range(step // steps_per_epoch):
        order = torch.randperm(len(images), generator=epoch_rng)
    cursor = (step % steps_per_epoch) * args.batch_size
    torch.manual_seed(args.seed + rank + step)
    running, started = 0.0, time.time()
    stop = args.steps if args.max_steps_this_run is None else min(args.steps, step + args.max_steps_this_run)
    progress = tqdm(total=stop, initial=step, disable=not main_process, desc="training")
    net.train()
    while step < stop:
        if cursor + args.batch_size > len(images):
            order, cursor = torch.randperm(len(images), generator=epoch_rng), 0
        index = order[cursor + rank * per_gpu: cursor + (rank + 1) * per_gpu].to(device)
        cursor += args.batch_size
        x = images[index].float() / 127.5 - 1
        flip = torch.rand(len(x), device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        t = torch.randint(0, args.timesteps, (len(x),), device=device)
        noisy, noise = diffusion.noise_images(x, t)
        lr = args.learning_rate * min(1.0, (step + 1) / args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = F.mse_loss(net(noisy, t).float(), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()
        ema.update(model)
        step += 1
        running += loss.detach()
        progress.update(1)

        if step % args.log_every == 0:
            mean = running / args.log_every
            if distributed:
                dist.all_reduce(mean)
                mean /= world
            if main_process:
                with (args.output_dir / "loss.csv").open("a", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow([step, f"{mean.item():.6f}", f"{lr:.3e}", f"{time.time() - started:.1f}"])
                progress.set_postfix(loss=f"{mean.item():.4f}")
            running = 0.0
        if main_process and (step % args.save_every == 0 or step == args.steps):
            save(args.output_dir / f"checkpoint_{step:06d}.pt")
        if main_process and (step % args.sample_every == 0 or step == args.steps):
            sampler = UNet(args.base_channels, dropout=args.dropout).to(device)
            sampler.load_state_dict(ema.shadow)
            sampler.eval()
            fixed = torch.Generator(device=device).manual_seed(args.seed)
            grid = diffusion.sample(sampler, args.sample_count, args.image_size, fixed)
            utils.save_image(grid, args.output_dir / f"samples_{step:06d}.png", nrow=8)
            del sampler
        if distributed and step % args.sample_every == 0:
            dist.barrier()

    if main_process and step == args.steps:
        torch.save({"ema": ema.shadow, "step": step, "class_counts": counts,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
                   args.output_dir / "final.pt")
        print(f"Finished {step} steps; EMA weights saved to {args.output_dir / 'final.pt'}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
