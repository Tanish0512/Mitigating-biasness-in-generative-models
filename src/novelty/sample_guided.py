"""Classifier-guided samples of one target class from an unconditional generator.

Same ancestral sampler as train_diffusion.Diffusion.sample, but the predicted
noise is shifted by the gradient of a noise-aware classifier
(train_noisy_classifier.py) towards the target class (Dhariwal & Nichol, 2021):

    eps_hat = eps(x_t, t) - s * sqrt(1 - alpha_bar_t) * grad_x log p(y | x_t, t)

The generator itself is unchanged; ``--scale`` s = 0 gives ordinary samples.
Images are written as ``<prefix>_<class>_<index>.png``. Each batch is seeded
from its start index and the class, so jobs can be split with ``--start`` and
``--count`` and still give the same images.

Example::

    python src/novelty/sample_guided.py --checkpoint ../runs/g0_v2/final.pt --classifier ../runs/noisy_classifier/final.pt \\
        --class-name butterfly --scale 3 --count 400 --output-dir ../results/guided_g0_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision import utils
from tqdm.auto import tqdm

# The shared modules live in the other src/ folders and in first_version/.
_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in (*sorted(_SRC.iterdir()), _SRC.parent / "first_version") if p.is_dir()]

from train_diffusion import Diffusion, UNet
from train_noisy_classifier import NoisyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Generator final.pt (EMA weights)")
    parser.add_argument("--classifier", type=Path, required=True, help="train_noisy_classifier.py final.pt")
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--prefix", default="guided")
    parser.add_argument("--seed", type=int, default=20261116)
    return parser.parse_args()


def guided_sample(diffusion: Diffusion, model: UNet, classifier: NoisyClassifier, target: int, scale: float,
                  n: int, image_size: int, generator: torch.Generator) -> torch.Tensor:
    device = diffusion.beta.device
    x = torch.randn(n, 3, image_size, image_size, device=device, generator=generator)
    y = torch.full((n,), target, device=device, dtype=torch.long)
    for step in range(diffusion.steps - 1, -1, -1):
        t = torch.full((n,), step, device=device, dtype=torch.long)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            eps = model(x, t).float()
        a = diffusion.alpha_bar[step]
        if scale:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    logits = classifier(x_in, t).float()
                log_p = logits.log_softmax(1)[torch.arange(n, device=device), y].sum()
                grad = torch.autograd.grad(log_p, x_in)[0]
            eps = eps - scale * (1 - a).sqrt() * grad
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        x = diffusion.mean_x0[step] * x0 + diffusion.mean_xt[step] * x
        if step > 0:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = x + diffusion.posterior_variance[step].sqrt() * noise
    return (x.clamp(-1, 1) + 1) / 2


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved = checkpoint["args"]
    model = UNet(saved["base_channels"], dropout=saved["dropout"]).to(device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    stored = torch.load(args.classifier, map_location="cpu", weights_only=False)
    classes = stored["classes"]
    classifier = NoisyClassifier(len(classes), stored["args"]["base_channels"]).to(device)
    classifier.load_state_dict(stored["model"])
    classifier.eval()
    target = classes.index(args.class_name)
    diffusion = Diffusion(saved["timesteps"], device, saved["schedule"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for begin in tqdm(range(args.start, args.start + args.count, args.batch_size), desc=f"{args.class_name} batches"):
        n = min(args.batch_size, args.start + args.count - begin)
        generator = torch.Generator(device=device).manual_seed(args.seed + 1_000_000 * target + begin)
        images = guided_sample(diffusion, model, classifier, target, args.scale, n, saved["image_size"], generator)
        for offset, image in enumerate(images.cpu()):
            utils.save_image(image, args.output_dir / f"{args.prefix}_{args.class_name}_{begin + offset:05d}.png")


if __name__ == "__main__":
    main()
