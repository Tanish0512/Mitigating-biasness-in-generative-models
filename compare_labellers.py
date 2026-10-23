"""Compare class labellers and density-bias metrics on dataset-0-v2 generations.

Every labeller is first checked on unseen real pool images, whose class is
known, then applied to the generated images:

* dinov2_probe: class-balanced logistic regression on DINOv2 features (as in
  evaluate_v2.py); ``+reject`` counts images with top probability < 0.5 as
  unrecognisable;
* clip_probe: the same probe on CLIP ViT-L/14 image features;
* clip_zeroshot: CLIP text prompts, forced to a class or with an extra
  "unrecognisable" option;
* resnet: the old 10-class ResNet-18, restricted to the 6 kept classes;
* consensus: a class only where the DINOv2 and CLIP probes agree, with
  probability >= 0.5 each; otherwise unrecognisable.

Class shares are taken over recognised images. Metrics against the training
shares p (q = generated shares, H/L = high/low group):

* HDAI = q_H / p_H. With p_H = 0.952 its maximum is 1.05.
* LDAI = q_L / p_L, from 0 (low classes gone) to 1 / p_L = 21.
* DOR, density odds ratio = (q_H / q_L) / (p_H / p_L): 1 = no bias, > 1 = the
  high group is favoured, unbounded; log2 DOR is symmetric around 0.
* worst low-class amplification, min over low classes of q_c / p_c;
* TVD and JSD (bits).

Label-free, per class in DINOv2 space (labels from the consensus labeller):
Vendi score (effective number of distinct images) of generated vs real images
of the class at equal sample size, and their ratio.

Example::

    python compare_labellers.py --data-dir ../data/dataset-0-v2 --pool-dir ../data/pool \\
        --generated G0v2=../results/samples_g0_v2:*.png --resnet-checkpoint ../data/classifier/checkpoint_best.pt \\
        --dino-cache ../results/eval_g0_v2/cache --output-dir ../results/labellers_g0_v2
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

import evaluate_clip
from evaluate_features import BACKBONE, Features
from evaluate_v2 import CLASSES, GROUPS, split, training_images, unseen_pool

UNRECOGNISABLE = "unrecognisable"
CLIP_NAMES = {
    "aircraft": ["airplane", "aircraft", "jet plane"],
    "car": ["car", "automobile"],
    "ship": ["ship", "boat"],
    "butterfly": ["butterfly"],
    "marine_animal": ["shark", "whale", "dolphin", "fish"],
    "human": ["person", "human face", "man or woman"],
}
OLD_RESNET_NAMES = {"marine_animal": "fish"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--generated", required=True, metavar="NAME=DIR:GLOB")
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-cache", type=Path, required=True, help="evaluate_v2.py's feature cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-count", type=int, default=4750)
    parser.add_argument("--vendi-size", type=int, default=40, help="Images per class for the Vendi score")
    parser.add_argument("--seed", type=int, default=20261022)
    return parser.parse_args()


class ClipFeatures:
    def __init__(self, device: torch.device, cache_dir: Path):
        self.model, self.processor, _ = evaluate_clip.load_clip(evaluate_clip.CLIP_MODEL, device)
        self.device, self.cache_dir = device, cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    @torch.inference_mode()
    def text(self) -> torch.Tensor:
        def embed(prompts):
            tokens = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            f = evaluate_clip._projected(self.model.get_text_features(**tokens))
            mean = (f / f.norm(dim=-1, keepdim=True)).mean(0)
            return mean / mean.norm()

        rows = [embed([t.format(n) for n in CLIP_NAMES[c] for t in evaluate_clip.TEMPLATES]) for c in CLASSES]
        rows.append(embed(evaluate_clip.UNRECOGNISABLE_PROMPTS))
        return torch.stack(rows).float().cpu()

    @torch.inference_mode()
    def __call__(self, name: str, paths: list[Path]) -> torch.Tensor:
        cache = self.cache_dir / f"clip_{name}.pt"
        keys = [str(p) for p in paths]
        if cache.is_file():
            stored = torch.load(cache, weights_only=True)
            if stored["paths"] == keys:
                return stored["features"]
        chunks = []
        for start in tqdm(range(0, len(paths), 256), desc=f"clip {name}"):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + 256]]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            f = evaluate_clip._projected(self.model.get_image_features(**inputs))
            chunks.append(torch.nn.functional.normalize(f, dim=-1).float().cpu())
        features = torch.cat(chunks)
        torch.save({"paths": keys, "features": features}, cache)
        return features


@torch.inference_mode()
def resnet_labels(paths: list[Path], checkpoint: Path, device: torch.device) -> list[str]:
    model, preprocess = evaluate_clip.load_resnet(checkpoint, device)
    keep = [evaluate_clip.CLASSES.index(OLD_RESNET_NAMES.get(c, c)) for c in CLASSES]
    labels = []
    for start in range(0, len(paths), 512):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + 512]]).to(device)
        labels += [CLASSES[i] for i in model(batch)[:, keep].argmax(1).tolist()]
    return labels


def probe(train_x: torch.Tensor, train_y: list[str]) -> LogisticRegression:
    model = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    return model.fit(train_x.numpy(), [CLASSES.index(c) for c in train_y])


def probe_labels(model: LogisticRegression, x: torch.Tensor, reject: float | None = None):
    probabilities = model.predict_proba(x.numpy())
    labels = [CLASSES[i] for i in probabilities.argmax(1)]
    if reject is not None:
        labels = [l if p >= reject else UNRECOGNISABLE for l, p in zip(labels, probabilities.max(1))]
    return labels, probabilities


def metrics(labels: list[str], p: dict[str, float]) -> dict:
    counts = Counter(labels)
    total = sum(counts[c] for c in CLASSES)
    q = {c: counts[c] / total for c in CLASSES}
    group_p = {g: sum(p[c] for c in m) for g, m in GROUPS.items()}
    group_q = {g: sum(q[c] for c in m) for g, m in GROUPS.items()}
    mix = {c: (p[c] + q[c]) / 2 for c in CLASSES}
    kl = lambda a, b: sum(a[c] * math.log2(a[c] / b[c]) for c in CLASSES if a[c] > 0)
    dor = (group_q["high"] / group_q["low"]) / (group_p["high"] / group_p["low"]) if group_q["low"] else math.inf
    return {
        "recognised": total,
        "unrecognisable_fraction": counts[UNRECOGNISABLE] / len(labels),
        "class_amplification": {c: q[c] / p[c] for c in CLASSES},
        "HDAI": group_q["high"] / group_p["high"],
        "LDAI": group_q["low"] / group_p["low"],
        "DOR": dor,
        "log2_DOR": math.log2(dor),
        "worst_low_amplification": min(q[c] / p[c] for c in GROUPS["low"]),
        "TVD": 0.5 * sum(abs(p[c] - q[c]) for c in CLASSES),
        "JSD_bits": 0.5 * kl(p, mix) + 0.5 * kl(q, mix),
    }


def vendi(features: torch.Tensor) -> float:
    """exp(entropy) of the eigenvalues of the cosine kernel / n (Friedman & Dieng, 2023)."""
    eigenvalues = torch.linalg.eigvalsh(features @ features.T / len(features)).clamp(min=0)
    eigenvalues = eigenvalues[eigenvalues > 0]
    return float(torch.exp(-(eigenvalues * eigenvalues.log()).sum()))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    training = training_images(args.data_dir)
    counts = Counter(c for _, c, _ in training)
    shares = {c: counts[c] / len(training) for c in CLASSES}
    reference, _ = split(training, args.baseline_count, shares, args.seed)
    ref_paths, ref_labels = [p for p, _, _ in reference], [c for _, c, _ in reference]
    used = {sha for _, _, sha in training}
    unseen = unseen_pool(args.pool_dir, used, {c: round(args.baseline_count * shares[c]) for c in CLASSES}, args.seed)
    unseen_paths, unseen_truth = [p for p, _ in unseen], [c for _, c in unseen]
    name, location = args.generated.split("=", 1)
    folder, pattern = location.rsplit(":", 1)
    gen_paths = sorted(Path(folder).glob(pattern))

    dino = Features(BACKBONE, device, 256, args.dino_cache)
    dino_x = {"ref": dino("train_reference", ref_paths), "unseen": dino("unseen_real", unseen_paths),
              "gen": dino(name, gen_paths)}
    del dino
    clip = ClipFeatures(device, args.output_dir / "cache")
    clip_x = {"ref": clip("train_reference", ref_paths), "unseen": clip("unseen_real", unseen_paths),
              "gen": clip(name, gen_paths)}
    text = clip.text()
    del clip

    dino_probe, clip_probe = probe(dino_x["ref"], ref_labels), probe(clip_x["ref"], ref_labels)
    labellers = {}
    for split_name in ("unseen", "gen"):
        d_forced, d_prob = probe_labels(dino_probe, dino_x[split_name])
        c_forced, c_prob = probe_labels(clip_probe, clip_x[split_name])
        logits = clip_x[split_name] @ text.T
        zero_open = [CLASSES[i] if i < len(CLASSES) else UNRECOGNISABLE for i in logits.argmax(1).tolist()]
        paths = unseen_paths if split_name == "unseen" else gen_paths
        consensus = [d if d == c and dp.max() >= 0.5 and cp.max() >= 0.5 else UNRECOGNISABLE
                     for d, c, dp, cp in zip(d_forced, c_forced, d_prob, c_prob)]
        labellers[split_name] = {
            "dinov2_probe": d_forced,
            "dinov2_probe+reject": probe_labels(dino_probe, dino_x[split_name], 0.5)[0],
            "clip_probe": c_forced,
            "clip_probe+reject": probe_labels(clip_probe, clip_x[split_name], 0.5)[0],
            "clip_zeroshot": [CLASSES[i] for i in logits[:, :len(CLASSES)].argmax(1).tolist()],
            "clip_zeroshot+unrecognisable": zero_open,
            "resnet_old": resnet_labels(paths, args.resnet_checkpoint, device),
            "consensus": consensus,
        }

    results = {}
    for labeller in labellers["gen"]:
        unseen_labels = labellers["unseen"][labeller]
        recognised = [(t, l) for t, l in zip(unseen_truth, unseen_labels) if l != UNRECOGNISABLE]
        results[labeller] = {
            "unseen_real_balanced_accuracy": float(balanced_accuracy_score(*zip(*recognised))),
            "unseen_real": metrics(unseen_labels, shares),
            name: metrics(labellers["gen"][labeller], shares),
        }

    # Vendi per class at equal sample size; generated images grouped by consensus label.
    diversity = {}
    gen_labels = labellers["gen"]["consensus"]
    for c in CLASSES:
        gen_idx = [i for i, l in enumerate(gen_labels) if l == c]
        real_idx = [i for i, l in enumerate(unseen_truth) if l == c]
        n = min(args.vendi_size, len(gen_idx), len(real_idx))
        if n < 10:
            diversity[c] = {"n": n}
            continue
        g = vendi(dino_x["gen"][rng.sample(gen_idx, n)])
        r = vendi(dino_x["unseen"][rng.sample(real_idx, n)])
        diversity[c] = {"n": n, "vendi_generated": g, "vendi_real": r, "ratio": g / r}

    (args.output_dir / "summary.json").write_text(json.dumps(
        {"training_shares": shares, "labellers": results, "vendi_per_class": diversity}, indent=2) + "\n")
    print(f"{'labeller':30s}{'acc':>6s}{'unrec':>7s}{'HDAI':>7s}{'LDAI':>7s}{'log2DOR':>9s}{'worstL':>8s}{'TVD':>7s}"
          f"   amplification {' '.join(c[:4] for c in CLASSES)}")
    for labeller, r in results.items():
        m = r[name]
        amp = " ".join(f"{m['class_amplification'][c]:4.2f}" for c in CLASSES)
        print(f"{labeller:30s}{r['unseen_real_balanced_accuracy']:6.3f}{m['unrecognisable_fraction']:7.1%}"
              f"{m['HDAI']:7.3f}{m['LDAI']:7.3f}{m['log2_DOR']:9.3f}{m['worst_low_amplification']:8.2f}"
              f"{m['TVD']:7.3f}   {amp}")
    print("unseen real, same labellers (should be HDAI 1, LDAI 1, log2DOR 0):")
    for labeller, r in results.items():
        m = r["unseen_real"]
        print(f"  {labeller:28s}HDAI {m['HDAI']:.3f}  LDAI {m['LDAI']:.3f}  log2DOR {m['log2_DOR']:+.3f}")
    print("Vendi per class (generated / real):", {c: round(d.get("ratio", float("nan")), 2) for c, d in diversity.items()})


if __name__ == "__main__":
    main()
