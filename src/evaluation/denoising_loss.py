"""Per-class denoising error of trained generators on held-out real D0 images.

Each model predicts the noise added to the same real images at the same timesteps with the same noise,
so differences come only from the weights. The error is reported per class and per noise level, which
shows whether loss weighting changes how well a model has learnt the high- and low-density classes.
The share of the training loss each class group receives under a weighting is printed alongside.

Example::

    python src/evaluation/denoising_loss.py --output ../results/denoising_loss.json \\
        --model G0=../runs/g0_v2/final.pt --model G0_inv=../runs/g0_v2_mit/final.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "generator")]
from train_diffusion import Diffusion, UNet

CLASSES = ["aircraft", "car", "ship", "butterfly", "marine_animal", "human"]
BASELINE = Path("../results/eval_all6/cache/train_baseline.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=75, help="Held-out real images per class")
    parser.add_argument("--timesteps", default="50,200,400,600,800,950")
    parser.add_argument("--repeats", type=int, default=2, help="Noise draws per image and timestep")
    parser.add_argument("--seed", type=int, default=20261121)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    rng = random.Random(args.seed)
    held_out = [Path(p) for p in torch.load(BASELINE, weights_only=True)["paths"]]
    images, classes = [], []
    for c in CLASSES:
        members = sorted(p for p in held_out if p.parent.name == c)
        for p in rng.sample(members, min(args.per_class, len(members))):
            images.append(np.asarray(Image.open(p).convert("RGB").resize((64, 64), Image.BICUBIC)))
            classes.append(c)
    x0 = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 127.5 - 1
    classes = np.array(classes)
    steps = [int(t) for t in args.timesteps.split(",")]
    generator = torch.Generator().manual_seed(args.seed)
    noises = [[torch.randn(x0.shape, generator=generator) for _ in range(args.repeats)] for _ in steps]

    results = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        saved = checkpoint["args"]
        model = UNet(saved["base_channels"], dropout=saved["dropout"])
        model.load_state_dict(checkpoint["ema"])
        model.eval()
        diffusion = Diffusion(saved["timesteps"], torch.device("cpu"), saved["schedule"])
        error = np.zeros((len(steps), len(x0)))
        for i, t in enumerate(steps):
            a = diffusion.alpha_bar[t]
            for noise in noises[i]:
                for begin in range(0, len(x0), 150):
                    x, n = x0[begin:begin + 150], noise[begin:begin + 150]
                    pred = model(a.sqrt() * x + (1 - a).sqrt() * n, torch.full((len(x),), t, dtype=torch.long))
                    error[i, begin:begin + 150] += (pred - n).square().flatten(1).mean(1).numpy() / args.repeats
        results[name] = {
            "weighting": saved.get("class_weighting", "none"),
            "per_class_mean": {c: round(float(error[:, classes == c].mean()), 5) for c in CLASSES},
            "per_class_by_t": {c: {str(t): round(float(error[i, classes == c].mean()), 5) for i, t in enumerate(steps)}
                               for c in CLASSES},
        }
        print(name, results[name]["per_class_mean"], flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"timesteps": steps, "images_per_class": args.per_class,
                                       "models": results}, indent=2) + "\n", encoding="utf-8")
    base = next(iter(results))
    print(f"\nchange in mean denoising error vs {base} (negative = learnt better)")
    print(f"{'model':10s}" + "".join(f"{c[:8]:>10s}" for c in CLASSES))
    for name, r in results.items():
        print(f"{name:10s}" + "".join(f"{100 * (r['per_class_mean'][c] / results[base]['per_class_mean'][c] - 1):+9.1f}%"
                                      for c in CLASSES))


if __name__ == "__main__":
    main()
