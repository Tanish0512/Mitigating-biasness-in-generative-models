"""Generate requested per-class images from the large class-conditional DDPM.

Used to top up D0's synthetic share. Images are written to
``<output-dir>/<class>/<prefix>_<index>.png`` with a manifest; screening and
selection happen later in build_dataset0_v2.py. ``--oversample`` generates
extra images so screened-out samples can be replaced.

Example::

    python generate_synthetic.py --checkpoint ../data/models/ddpm_checkpoint_0550.pt \\
        --counts aircraft=209 car=204 ship=1150 fish=10 human=10 --output-dir ../data/synthetic_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath

import torch
from torchvision import utils
from tqdm.auto import tqdm

# Alphabetical ImageFolder order the large DDPM was trained with.
CLASSES = ["aircraft", "bird", "butterfly", "car", "cat", "dog", "fish", "flower", "human", "ship"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--counts", nargs="+", required=True, metavar="CLASS=N")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oversample", type=float, default=1.25)
    parser.add_argument("--prefix", default="synv2")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


def load_conditional(path: Path, device: torch.device):
    from train import Diffusion, UNet

    torch.serialization.add_safe_globals([(PurePosixPath, "pathlib.PosixPath")])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    if "class_embedding.weight" not in state:
        raise ValueError("Checkpoint is unconditional; a class-conditional DDPM is required")
    if list(checkpoint.get("classes", CLASSES)) != CLASSES:
        raise ValueError(f"Checkpoint class order {checkpoint.get('classes')} does not match {CLASSES}")
    args = checkpoint.get("args", {})
    model = UNet(num_classes=len(CLASSES), base_channels=int(args.get("base_channels", 64))).to(device)
    model.load_state_dict(checkpoint.get("ema", state))
    return model.eval(), Diffusion(int(args.get("timesteps", 1000)), device), int(args.get("image_size", 64)), args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = {}
    for item in args.counts:
        name, value = item.split("=")
        if name not in CLASSES:
            raise ValueError(f"Unknown class {name}")
        counts[name] = math.ceil(int(value) * args.oversample)
    torch.manual_seed(args.seed)
    model, diffusion, image_size, saved_args = load_conditional(args.checkpoint, device)

    labels = [CLASSES.index(name) for name, n in counts.items() for _ in range(n)]
    written = {name: 0 for name in counts}
    rows = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(labels), args.batch_size), desc="generating"):
            batch = torch.tensor(labels[start:start + args.batch_size], device=device)
            for label, image in zip(batch.tolist(), diffusion.sample(model, len(batch), batch, image_size).cpu()):
                name = CLASSES[label]
                folder = args.output_dir / name
                folder.mkdir(parents=True, exist_ok=True)
                filename = f"{args.prefix}_{written[name]:05d}.png"
                utils.save_image(image, folder / filename)
                rows.append([filename, name])
                written[name] += 1

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "requested_class"])
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": saved_args.get("epochs"),
        "seed": args.seed,
        "oversample": args.oversample,
        "generated_per_class": written,
        "image_size": image_size,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {sum(written.values())} images: {written}")


if __name__ == "__main__":
    main()
