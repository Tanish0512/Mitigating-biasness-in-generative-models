"""Generate unconditional samples from a train_diffusion.py ``final.pt`` (EMA weights).

Several processes can share the work: each writes ``<prefix>_<index>.png`` for
its own index range, so ``--start`` and ``--count`` split one seeded job.

Example::

    python sample_diffusion.py --checkpoint runs/g0_v2/final.pt --output-dir samples/g0_v2 --start 0 --count 950
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import utils
from tqdm.auto import tqdm

from train_diffusion import Diffusion, UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--prefix", default="sample")
    parser.add_argument("--seed", type=int, default=20261023)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved = checkpoint["args"]
    model = UNet(saved["base_channels"], dropout=saved["dropout"]).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    diffusion = Diffusion(saved["timesteps"], device, saved["schedule"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for begin in tqdm(range(args.start, args.start + args.count, args.batch_size), desc="batches"):
        n = min(args.batch_size, args.start + args.count - begin)
        # Seed per batch start, so the images do not depend on how the job is split.
        generator = torch.Generator(device=device).manual_seed(args.seed + begin)
        for offset, image in enumerate(diffusion.sample(model, n, saved["image_size"], generator).cpu()):
            utils.save_image(image, args.output_dir / f"{args.prefix}_{begin + offset:05d}.png")


if __name__ == "__main__":
    main()
