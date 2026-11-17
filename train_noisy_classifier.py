"""Train a noise-aware classifier p(y | x_t, t) for classifier-guided sampling.

The classifier sees D0 images noised with the same cosine schedule as the
generators, at a random timestep, and predicts the class folder. It is the
encoder half of the generator's U-Net (time-conditioned residual blocks,
64 -> 32 -> 16 -> 8, attention at 8x8) followed by global average pooling and a
linear head. Batches are class-balanced, because D0 is 20:1 between high and
low classes. 10% of every class is held out to report accuracy per noise level.

Example::

    python train_noisy_classifier.py --data-dir ../data/dataset-0-v2 --output-dir ../runs/noisy_classifier
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from train_diffusion import Block, Diffusion, Down, SelfAttention, load_images, sinusoidal_embedding


class NoisyClassifier(nn.Module):
    """Time-conditioned encoder: 64 -> 32 -> 16 -> 8, attention at 8x8, pooled linear head."""

    def __init__(self, classes: int, base_channels: int = 64, time_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        c = base_channels
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input = nn.Conv2d(3, c, 3, padding=1)
        self.down1 = Down(c, c * 2, time_dim, dropout)
        self.down2 = Down(c * 2, c * 4, time_dim, dropout)
        self.down3 = Down(c * 4, c * 4, time_dim, dropout)
        self.mid = Block(c * 4, c * 4, time_dim, dropout)
        self.attention = SelfAttention(c * 4, time_dim)
        self.head = nn.Sequential(nn.GroupNorm(8, c * 4), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(c * 4, classes))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(sinusoidal_embedding(timesteps, self.time_mlp[0].in_features))
        x = self.input(x)
        for down in (self.down1, self.down2, self.down3):
            _, x = down(x, t)
        return self.head(self.attention(self.mid(x, t), t))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--evaluate-every", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20261116)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: NoisyClassifier, diffusion: Diffusion, images: torch.Tensor, labels: torch.Tensor,
             classes: int, seed: int) -> dict:
    """Balanced accuracy on held-out images at fixed noise levels (fixed noise, so runs are comparable)."""
    model.eval()
    generator = torch.Generator(device=images.device).manual_seed(seed)
    result = {}
    for step in (0, 100, 250, 500, 750, 900, 999):
        correct, total = torch.zeros(classes), torch.zeros(classes)
        for begin in range(0, len(images), 512):
            x = images[begin:begin + 512].float() / 127.5 - 1
            y = labels[begin:begin + 512]
            t = torch.full((len(x),), step, device=x.device, dtype=torch.long)
            noise = torch.randn(x.shape, device=x.device, generator=generator)
            a = diffusion.alpha_bar[t][:, None, None, None]
            with torch.autocast(x.device.type, dtype=torch.bfloat16):
                predicted = model(a.sqrt() * x + (1 - a).sqrt() * noise, t).argmax(1)
            correct += torch.bincount(y[predicted == y], minlength=classes).cpu()
            total += torch.bincount(y, minlength=classes).cpu()
        result[str(step)] = round((correct / total).mean().item(), 4)
    model.train()
    return result


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    images, counts = load_images(args.data_dir, args.image_size)
    labels = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(list(counts.values())))
    classes = len(counts)
    # Per-class random split; the held-out part is never trained on.
    split = torch.Generator().manual_seed(args.seed)
    train_index, val_index = [], []
    for c in range(classes):
        members = (labels == c).nonzero().squeeze(1)
        members = members[torch.randperm(len(members), generator=split)]
        n_val = round(len(members) * args.validation_fraction)
        val_index.append(members[:n_val])
        train_index.append(members[n_val:])
    images, labels = images.to(device), labels.to(device)
    val_images, val_labels = images[torch.cat(val_index)], labels[torch.cat(val_index)]
    # Training indices grouped by class, so a class-balanced batch is one gather.
    flat = torch.cat(train_index).to(device)
    sizes = torch.tensor([len(i) for i in train_index], device=device)
    starts = torch.cumsum(sizes, 0) - sizes

    model = NoisyClassifier(classes, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, total_steps=args.steps,
                                                   pct_start=0.05)
    diffusion = Diffusion(args.timesteps, device, args.schedule)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    record.update(classes=list(counts), class_counts=counts, parameters=sum(p.numel() for p in model.parameters()),
                  train_images=sum(len(i) for i in train_index), validation_images=len(val_images), evaluations=[])
    print(f"{record['parameters']:,} parameters; train {record['train_images']}, validation {len(val_images)}")

    started = time.time()
    progress = tqdm(range(1, args.steps + 1), desc="training")
    for step in progress:
        # Class-balanced batch: a uniform class, then a uniform image of that class.
        y = torch.randint(0, classes, (args.batch_size,), device=device)
        index = flat[starts[y] + (torch.rand(len(y), device=device) * sizes[y]).long()]
        x = images[index].float() / 127.5 - 1
        flip = torch.rand(len(x), device=device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        t = torch.randint(0, args.timesteps, (len(x),), device=device)
        noisy, _ = diffusion.noise_images(x, t)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            loss = F.cross_entropy(model(noisy, t).float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 100 == 0:
            progress.set_postfix(loss=f"{loss.item():.3f}")
        if step % args.evaluate_every == 0 or step == args.steps:
            accuracy = evaluate(model, diffusion, val_images, val_labels, classes, args.seed)
            record["evaluations"].append({"step": step, "seconds": round(time.time() - started),
                                          "balanced_accuracy_by_t": accuracy})
            print(f"step {step}: balanced accuracy by t {accuracy}", flush=True)
            (args.output_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    torch.save({"model": model.state_dict(), "classes": list(counts), "args": record}, args.output_dir / "final.pt")
    print(f"Saved {args.output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
