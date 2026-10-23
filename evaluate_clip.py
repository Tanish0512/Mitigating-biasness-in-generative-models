"""Measure the class/density bias of generated images with a CLIP zero-shot judge.

CLIP was never trained on project data, so it gives an independent view of
which class each generated image shows. Each image is classified twice:

* forced: the best of the ten object classes, comparable to the ResNet;
* open: the ten classes plus "unrecognisable", so images that show no clear
  object are counted separately instead of being forced into a class.

Images come either from an existing folder (for example G0's samples inside
dataset-1) or are generated from an unconditional G0/G1 checkpoint. The
ResNet-18 classifier can be added as a second opinion.

Metrics follow PROJECT_LOG.md: class amplification q_i / p_i, group
amplification (HDAI is the high-density group), TVD and JSD in bits, all
against a reference class distribution p (D0 by default).

Examples::

    # G0's 4,750 unfiltered samples stored in dataset-1
    python evaluate_clip.py --image-dir ../data/d1/dataset-1 --glob "*/g0_sample_*.png" \\
        --resnet-checkpoint ../data/classifier/checkpoint_best.pt --output-dir eval_g0

    # Generate 2,000 images from G1 and evaluate them against D1's composition
    python evaluate_clip.py --generator-checkpoint ../data/models/g1.pt --num-samples 2000 \\
        --reference-summary ../data/d1/dataset-1/summary.json --output-dir eval_g1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path, PurePosixPath

import torch
from PIL import Image
from torchvision import transforms, utils
from tqdm.auto import tqdm


CLASSES = ["aircraft", "bird", "butterfly", "car", "cat", "dog", "fish", "flower", "human", "ship"]
GROUPS = {
    "high": ["aircraft", "bird", "car"],
    "medium": ["butterfly", "cat", "dog"],
    "low": ["fish", "flower", "human", "ship"],
}
# D0 class counts (README): 5,000 high, 1,000 medium, 250 low per class.
D0_COUNTS = {name: 5000 if name in GROUPS["high"] else 1000 if name in GROUPS["medium"] else 250 for name in CLASSES}
UNRECOGNISABLE = "unrecognisable"

# Several names per class; each is expanded with every template and averaged.
CLASS_NAMES = {
    "aircraft": ["airplane", "aircraft", "jet plane"],
    "bird": ["bird"],
    "butterfly": ["butterfly"],
    "car": ["car", "automobile"],
    "cat": ["cat"],
    "dog": ["dog"],
    "fish": ["fish"],
    "flower": ["flower"],
    "human": ["person", "human", "man or woman"],
    "ship": ["ship", "boat"],
}
TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low resolution photo of a {}.",
    "a pixelated photo of a {}.",
    "a small photo of a {}.",
]
UNRECOGNISABLE_PROMPTS = [
    "a blurry image with no recognisable object.",
    "an abstract pattern of colours.",
    "a meaningless blob of colours.",
    "random visual noise.",
    "a smeared, distorted image of nothing in particular.",
]
CLIP_MODEL = "openai/clip-vit-large-patch14"
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir", type=Path, help="Folder of images to evaluate (searched recursively)")
    source.add_argument("--generator-checkpoint", type=Path, help="Unconditional G0/G1 checkpoint to sample from")
    parser.add_argument("--glob", default="**/*.png", help="Pattern under --image-dir")
    parser.add_argument("--num-samples", type=int, default=1000, help="Images to generate with --generator-checkpoint")
    parser.add_argument("--max-images", type=int, default=None, help="Evaluate a random subset of --image-dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, default=None,
                        help="summary.json whose class counts form the reference p (default: D0 counts)")
    parser.add_argument("--reference-key", default="final_class_folder_counts")
    parser.add_argument("--resnet-checkpoint", type=Path, default=None, help="Optional ResNet-18 second opinion")
    parser.add_argument("--clip-model", default=CLIP_MODEL)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=64, help="Images per class in the sample grids")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(requested)


def load_checkpoint(path: Path) -> dict:
    # Safe loading; Linux-saved PosixPath arguments become PurePosixPath on any OS.
    torch.serialization.add_safe_globals([(PurePosixPath, "pathlib.PosixPath")])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    return checkpoint


def generate_images(path: Path, count: int, batch_size: int, output_dir: Path, device: torch.device) -> list[Path]:
    from train import Diffusion, UNet

    checkpoint = load_checkpoint(path)
    state = checkpoint.get("model")
    if not isinstance(state, dict) or "class_embedding.weight" in state:
        raise ValueError("Generator checkpoint must contain an unconditional model")
    args = checkpoint.get("args", {})
    image_size = int(args.get("image_size", 64))
    model = UNet(num_classes=0, base_channels=int(args.get("base_channels", 32))).to(device)
    model.load_state_dict(checkpoint.get("ema", state))
    model.eval()
    diffusion = Diffusion(int(args.get("timesteps", 1000)), device)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with torch.inference_mode():
        for start in tqdm(range(0, count, batch_size), desc=f"sampling {path.name}"):
            samples = diffusion.sample(model, min(batch_size, count - start), None, image_size).cpu()
            for offset, sample in enumerate(samples):
                image_path = output_dir / f"sample_{start + offset:05d}.png"
                utils.save_image(sample, image_path)
                paths.append(image_path)
    return paths


def _projected(output) -> torch.Tensor:
    # transformers >= 5 returns a model output whose pooler_output holds the projection.
    return output if isinstance(output, torch.Tensor) else output.pooler_output


def load_clip(name: str, device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(name)

    def embed(prompts: list[str]) -> torch.Tensor:
        tokens = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        features = _projected(model.get_text_features(**tokens))
        features = features / features.norm(dim=-1, keepdim=True)
        mean = features.mean(0)
        return mean / mean.norm()

    with torch.inference_mode():
        class_embeddings = [embed([t.format(n) for n in CLASS_NAMES[c] for t in TEMPLATES]) for c in CLASSES]
        class_embeddings.append(embed(UNRECOGNISABLE_PROMPTS))
    return model, processor, torch.stack(class_embeddings)


def load_resnet(path: Path, device: torch.device) -> tuple[torch.nn.Module, transforms.Compose]:
    from train_resnet import build_model

    checkpoint = load_checkpoint(path)
    if list(checkpoint.get("classes", ())) != CLASSES:
        raise ValueError(f"ResNet class order {checkpoint.get('classes')} does not match {CLASSES}")
    size = int(checkpoint.get("args", {}).get("image_size", 64))
    model = build_model(len(CLASSES), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(RESNET_MEAN, RESNET_STD),
    ])
    return model.to(device).eval(), preprocess


@torch.inference_mode()
def classify(paths: list[Path], args: argparse.Namespace, device: torch.device) -> list[dict]:
    clip, processor, text = load_clip(args.clip_model, device)
    resnet = load_resnet(args.resnet_checkpoint, device) if args.resnet_checkpoint else None
    rows = []
    for start in tqdm(range(0, len(paths), args.batch_size), desc="classifying"):
        batch_paths = paths[start:start + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        features = _projected(clip.get_image_features(pixel_values=pixels))
        features = features / features.norm(dim=-1, keepdim=True)
        logits = clip.logit_scale.exp() * features @ text.T
        forced = logits[:, :len(CLASSES)].softmax(1).cpu()
        open_set = logits.softmax(1).cpu()
        if resnet:
            model, preprocess = resnet
            resnet_probs = model(torch.stack([preprocess(i) for i in images]).to(device)).softmax(1).cpu()
        for i, path in enumerate(batch_paths):
            open_index = int(open_set[i].argmax())
            row = {
                "path": str(path),
                "clip_forced": CLASSES[int(forced[i].argmax())],
                "clip_forced_confidence": float(forced[i].max()),
                "clip_open": (CLASSES + [UNRECOGNISABLE])[open_index],
                "clip_open_confidence": float(open_set[i, open_index]),
            }
            if resnet:
                row["resnet"] = CLASSES[int(resnet_probs[i].argmax())]
                row["resnet_confidence"] = float(resnet_probs[i].max())
            rows.append(row)
    return rows


def distribution_metrics(counts: Counter, reference: dict[str, float]) -> dict:
    """Class/group amplification, HDAI, TVD and JSD (bits) of counts against reference p."""
    total = sum(counts[c] for c in CLASSES)
    if total == 0:
        raise ValueError("No images were assigned to an object class")
    q = {c: counts[c] / total for c in CLASSES}
    p = reference
    mix = {c: (p[c] + q[c]) / 2 for c in CLASSES}

    def kl(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(a[c] * math.log2(a[c] / b[c]) for c in CLASSES if a[c] > 0)

    groups = {
        g: sum(q[c] for c in members) / sum(p[c] for c in members) for g, members in GROUPS.items()
    }
    return {
        "count": total,
        "class_counts": {c: counts[c] for c in CLASSES},
        "class_shares": q,
        "class_amplification": {c: q[c] / p[c] for c in CLASSES},
        "group_shares": {g: sum(q[c] for c in members) for g, members in GROUPS.items()},
        "group_amplification": groups,
        "HDAI": groups["high"],
        "TVD": 0.5 * sum(abs(p[c] - q[c]) for c in CLASSES),
        "JSD_bits": 0.5 * kl(p, mix) + 0.5 * kl(q, mix),
    }


def load_reference(args: argparse.Namespace) -> tuple[dict[str, float], str]:
    if args.reference_summary is None:
        counts, source = D0_COUNTS, "D0 class counts"
    else:
        summary = json.loads(args.reference_summary.read_text(encoding="utf-8"))
        counts, source = summary[args.reference_key], f"{args.reference_summary}:{args.reference_key}"
    total = sum(counts[c] for c in CLASSES)
    return {c: counts[c] / total for c in CLASSES}, source


def save_grids(rows: list[dict], key: str, output_dir: Path, per_class: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    rng = random.Random(seed)
    for label in CLASSES + [UNRECOGNISABLE]:
        members = [r["path"] for r in rows if r.get(key) == label]
        if not members:
            continue
        chosen = rng.sample(members, min(per_class, len(members)))
        tensors = [to_tensor(Image.open(p).convert("RGB")) for p in chosen]
        utils.save_image(torch.stack(tensors), output_dir / f"{label}.png", nrow=8, padding=2)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.generator_checkpoint:
        paths = generate_images(args.generator_checkpoint, args.num_samples, args.sample_batch_size,
                                args.output_dir / "samples", device)
    else:
        paths = sorted(args.image_dir.glob(args.glob))
        if args.max_images and len(paths) > args.max_images:
            paths = sorted(random.Random(args.seed).sample(paths, args.max_images))
    if not paths:
        raise FileNotFoundError("No images to evaluate")

    reference, reference_source = load_reference(args)
    rows = classify(paths, args, device)

    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    open_counts = Counter(r["clip_open"] for r in rows)
    summary = {
        "images": len(rows),
        "source": str(args.generator_checkpoint or args.image_dir),
        "glob": None if args.generator_checkpoint else args.glob,
        "clip_model": args.clip_model,
        "seed": args.seed,
        "reference": reference_source,
        "reference_shares": reference,
        "clip_forced": distribution_metrics(Counter(r["clip_forced"] for r in rows), reference),
        # Shares over recognised images only; the unrecognisable count is reported beside them.
        "clip_open": {
            "unrecognisable_count": open_counts[UNRECOGNISABLE],
            "unrecognisable_fraction": open_counts[UNRECOGNISABLE] / len(rows),
            **distribution_metrics(open_counts, reference),
        },
    }
    if args.resnet_checkpoint:
        summary["resnet_checkpoint"] = str(args.resnet_checkpoint)
        summary["resnet"] = distribution_metrics(Counter(r["resnet"] for r in rows), reference)
        summary["clip_resnet_agreement"] = sum(r["resnet"] == r["clip_forced"] for r in rows) / len(rows)
        summary["clip_vs_resnet_counts"] = {
            c: dict(Counter(r["resnet"] for r in rows if r["clip_forced"] == c)) for c in CLASSES
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_grids(rows, "clip_open", args.output_dir / "grids_clip_open", args.grid_size, args.seed)
    if args.resnet_checkpoint:
        save_grids(rows, "resnet", args.output_dir / "grids_resnet", args.grid_size, args.seed)

    print(f"Evaluated {len(rows)} images against {reference_source}")
    for key in ("clip_forced", "clip_open", "resnet"):
        if key in summary:
            m = summary[key]
            print(f"{key:12s} HDAI={m['HDAI']:.4f} TVD={m['TVD']:.4f} JSD={m['JSD_bits']:.4f} bits "
                  f"groups={ {g: round(v, 3) for g, v in m['group_amplification'].items()} }")
    print(f"unrecognisable: {summary['clip_open']['unrecognisable_fraction']:.1%}")
    if "clip_resnet_agreement" in summary:
        print(f"CLIP/ResNet agreement: {summary['clip_resnet_agreement']:.1%}")


if __name__ == "__main__":
    main()
