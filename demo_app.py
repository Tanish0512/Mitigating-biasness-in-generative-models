"""Test interface for the project: generate live from the six trained models and measure their bias.

Tabs:
  1. Live generation - pick models, image count and seed; every model starts from the same noise,
     the labeller v2 counts the classes, and the bias metrics are compared with D0.
  2. Full evaluation - the stored 4,750-image results for all six models, with a browser of the
     images each model drew for a chosen class.
  3. Audit images - upload any images (real test images or generated ones) and label them.
  4. Pipeline - what each dataset and model is.

Live images are fresh draws: 1000 denoising steps take about 60 s per 100 images on an A6000, so
keep live runs small and use the Full evaluation tab for the 4,750-image numbers.

Example::

    CUDA_DEVICE_ORDER=PCI_BUS_ID python demo_app.py --gpus 3,4,0,1,5,3 --labeller-gpu 2
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import evaluate_clip
from evaluate_features import BACKBONE
from evaluate_v2 import CLASSES, GROUPS
from train_diffusion import Diffusion, UNet

ROOT = Path(__file__).resolve().parent.parent
BACKGROUND, UNCOUNTED = "background", "unrecognisable"
D0_COUNTS = {"aircraft": 6000, "car": 6000, "ship": 6000, "butterfly": 300, "marine_animal": 300, "human": 300}
D0_SHARES = {c: n / sum(D0_COUNTS.values()) for c, n in D0_COUNTS.items()}
EVAL_IMAGES = 4750
SAMPLE_SEED = 20261023
SAMPLE_BATCH = 250  # seeded per batch like sample_diffusion.py

# name: (run, stored sample folder, stored label file, what it is)
MODELS = {
    "G0 (baseline)": ("g0_v2", "samples_g0_v2", "repo/results_g0_v2/classifier_new_p0.9/labels_G0v2.csv",
                      "trained on D0 (real + 20% conditional-DDPM images)"),
    "G1, 25% mixing": ("g1_v2", "samples_g1_v2", "repo/results_g1_v2/classifier_new_p0.9/labels_G1v2.csv",
                       "trained on D1 = 75% D0 + 25% unfiltered G0 samples"),
    "G1, 50% mixing": ("g1_v2_r50", "samples_g1_v2_r50", "repo/results_g1_v2_r50/labels_G1v2r50.csv",
                       "trained on D1 = 50% D0 + 50% unfiltered G0 samples"),
    "G0 mitigated (weighted loss)": ("g0_v2_mit", "samples_g0_v2_mit", "repo/results_mitigation/g0_mit/labels_G0v2mit.csv",
                                     "D0 with class-weighted loss (w ~ n^-1/2)"),
    "G1 mitigated (curated + weighted)": ("g1_v2_mit", "samples_g1_v2_mit", "repo/results_mitigation/g1_mit/labels_G1v2mit.csv",
                                          "curated 25% D1 with class-weighted loss"),
    "G1 curated mixing only": ("g1_v2_curated", "samples_g1_v2_curated", "repo/results_mitigation/g1_curated/labels_G1v2cur.csv",
                               "curated 25% D1 (confident G0 samples in D0 ratios), plain loss"),
}
DEFAULT_LIVE = ["G0 (baseline)", "G1, 25% mixing", "G1 mitigated (curated + weighted)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="0", help="comma-separated GPUs for the generators, used in turn")
    parser.add_argument("--labeller-gpu", type=int, default=0)
    parser.add_argument("--labeller", type=Path, default=ROOT / "results/labeller_v2/labeller.pkl")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="also create a public gradio.live link")
    return parser.parse_args()


class Generator:
    def __init__(self, run: str, device: torch.device):
        checkpoint = torch.load(ROOT / "runs" / run / "final.pt", map_location="cpu", weights_only=True)
        saved = checkpoint["args"]
        self.model = UNet(saved["base_channels"], dropout=saved["dropout"]).to(device).eval()
        self.model.load_state_dict(checkpoint["ema"])
        self.diffusion = Diffusion(saved["timesteps"], device, saved["schedule"])
        self.image_size, self.device = saved["image_size"], device
        self.lock = threading.Lock()

    @torch.inference_mode()
    def __call__(self, n: int, seed: int) -> list[Image.Image]:
        images = []
        with self.lock:
            for begin in range(0, n, SAMPLE_BATCH):
                generator = torch.Generator(device=self.device).manual_seed(seed + begin)
                batch = self.diffusion.sample(self.model, min(SAMPLE_BATCH, n - begin), self.image_size, generator)
                images += [to_pil(x) for x in batch.cpu()]
        return images


def to_pil(x: torch.Tensor) -> Image.Image:
    # The same rounding as torchvision.utils.save_image, so live labels match the stored PNGs.
    return Image.fromarray(x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8).numpy())


class Labeller:
    """Labeller v2: logistic regression on DINOv2 + CLIP features (see train_labeller.py)."""

    def __init__(self, path: Path, device: torch.device):
        from transformers import AutoImageProcessor, AutoModel

        with path.open("rb") as handle:
            stored = pickle.load(handle)
        self.model, self.labels = stored["model"], stored["labels"]
        self.dino = AutoModel.from_pretrained(BACKBONE).to(device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(BACKBONE)
        self.clip, self.clip_processor, _ = evaluate_clip.load_clip(evaluate_clip.CLIP_MODEL, device)
        self.device = device
        self.lock = threading.Lock()

    @torch.inference_mode()
    def __call__(self, images: list[Image.Image]) -> list[tuple[str, float]]:
        features = []
        with self.lock:
            for start in range(0, len(images), 128):
                chunk = [im.convert("RGB") for im in images[start:start + 128]]
                inputs = self.dino_processor(images=chunk, return_tensors="pt").to(self.device)
                dino = torch.nn.functional.normalize(self.dino(**inputs).last_hidden_state[:, 0], dim=-1)
                inputs = self.clip_processor(images=chunk, return_tensors="pt").to(self.device)
                clip = torch.nn.functional.normalize(evaluate_clip._projected(self.clip.get_image_features(**inputs)), dim=-1)
                features.append(torch.cat([dino, clip], 1).float().cpu())
        probabilities = self.model.predict_proba(torch.cat(features).numpy())
        return [(self.labels[p.argmax()], float(p.max())) for p in probabilities]


def counted(label: str, probability: float, threshold: float) -> str:
    return label if probability >= threshold and label != BACKGROUND else UNCOUNTED


def bias_metrics(labels: list[str]) -> dict:
    """The report's metrics for one set of counted labels, against the D0 class shares."""
    counts = Counter(labels)
    total = sum(counts[c] for c in CLASSES)
    row = {c: counts[c] for c in CLASSES}
    low = sum(counts[c] for c in GROUPS["low"])
    row.update({"images": len(labels), "low total": low,
                "expected low": round(total * sum(D0_SHARES[c] for c in GROUPS["low"]), 1),
                "not counted %": round(100 * counts[UNCOUNTED] / max(len(labels), 1), 1)})
    if total == 0:
        return row | {"HDAI": float("nan"), "log2 DOR": float("nan"), "worst low": float("nan"), "TVD": float("nan")}
    q = {c: counts[c] / total for c in CLASSES}
    qh, ph = sum(q[c] for c in GROUPS["high"]), sum(D0_SHARES[c] for c in GROUPS["high"])
    dor = (qh / (1 - qh)) / (ph / (1 - ph)) if qh < 1 else math.inf
    row.update({"HDAI": round(qh / ph, 3), "log2 DOR": round(math.log2(dor), 2),
                "worst low": round(min(q[c] / D0_SHARES[c] for c in GROUPS["low"]), 2),
                "TVD": round(0.5 * sum(abs(q[c] - D0_SHARES[c]) for c in CLASSES), 3)})
    return row


TABLE_COLUMNS = ["model", "images", *CLASSES, "low total", "expected low", "not counted %", "log2 DOR", "HDAI", "worst low", "TVD"]
LIVE_COLUMNS = [*TABLE_COLUMNS, "log2 DOR, stored 4,750"]


def table(rows: dict[str, dict]) -> list[list]:
    return [[name, *[r[k] for k in TABLE_COLUMNS[1:]]] for name, r in rows.items()]


def share_chart(rows: dict[str, dict], title: str):
    """Class share among counted images, one group of bars per class; the black tick is D0's share."""
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=110)
    names = list(rows)
    width = 0.8 / max(len(names), 1)
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B3", "#937860"]
    for i, name in enumerate(names):
        r = rows[name]
        total = sum(r[c] for c in CLASSES) or 1
        shares = [100 * r[c] / total for c in CLASSES]
        ax.bar(np.arange(len(CLASSES)) + (i - (len(names) - 1) / 2) * width, shares, width * 0.92,
               label=name, color=colors[list(MODELS).index(name) % len(colors)])
    for j, c in enumerate(CLASSES):
        ax.hlines(100 * D0_SHARES[c], j - 0.45, j + 0.45, colors="black", linewidth=2)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30, 100], ["0.1", "0.3", "1", "3", "10", "30", "100"])
    ax.set_xticks(range(len(CLASSES)), [c.replace("_", " ") for c in CLASSES])
    ax.set_ylabel("% of counted images (log)")
    ax.set_title(title + "   (black line = D0 share)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    return fig


def stored_results() -> tuple[dict[str, dict], dict[str, list[tuple[str, str, float]]]]:
    """Counted labels (p >= 0.9) for the first 4,750 stored samples of each model."""
    rows, labelled = {}, {}
    for name, (_, folder, label_file, _) in MODELS.items():
        with (ROOT / label_file).open(newline="") as handle:
            records = sorted(csv.DictReader(handle), key=lambda r: r["image"])[:EVAL_IMAGES]
        labelled[name] = [(str(ROOT / "results" / folder / r["image"]), r["label"], float(r["probability"])) for r in records]
        rows[name] = bias_metrics([counted(l, p, 0.9) for _, l, p in labelled[name]])
    return rows, labelled


PIPELINE_MD = f"""
### Pipeline: D0 → G0 → D1 → G1

| Dataset | aircraft / car / ship | butterfly / marine animal / human | Notes |
|---|---|---|---|
| **D0** | 6,000 each | 300 each | 18,900 images, 20:1 high:low, 20% synthetic in every class |
| **D1, 25%** | 4,500 real + 4,725 G0 samples (unfiltered) | 225 real each | G0 decides the class mix of the synthetic part |
| **D1, 50%** | 3,000 real + 9,450 G0 samples (unfiltered) | 150 real each | twice as much feedback |
| **D1, curated** | 4,500 real + 1,500 confident G0 each | 225 real + 75 confident G0 each | exactly D0's class ratio |

All generators are the same unconditional DDPM (22.4M-parameter U-Net, cosine schedule, 150k steps, EMA).
Class labels are never given to the generators.

**Bias metrics** (on images the labeller is ≥ threshold confident about):
- **log2 DOR** = log2[(q_high/q_low) / (p_high/p_low)]: 0 = no bias, +1 = high classes favoured 2×, negative = low classes favoured.
- **HDAI** = q_high/p_high: its maximum is {1 / sum(D0_SHARES[c] for c in GROUPS['high']):.2f} here, so it barely moves.
- **worst low** = min over low classes of q_c/p_c (1 = class kept its share, 0 = class vanished).
- **TVD**: total variation distance between generated and D0 class shares.

**Mitigations**: (1) class-weighted loss w_c ∝ n_c^-1/2 (high 0.86, low 3.84); (2) curated mixing:
feed back only confident G0 samples in D0's class ratios.
"""


def main() -> None:
    args = parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    print("loading generators ...")
    generators = {name: Generator(run, torch.device(f"cuda:{gpus[i % len(gpus)]}"))
                  for i, (name, (run, *_)) in enumerate(MODELS.items())}
    print("loading labeller ...")
    labeller = Labeller(args.labeller, torch.device(f"cuda:{args.labeller_gpu}"))
    stored_rows, stored_labels = stored_results()

    def generate(selected, n, seed, threshold, progress=gr.Progress()):
        if not selected:
            raise gr.Error("Pick at least one model.")
        n, seed = int(n), int(seed)
        progress(0, desc=f"sampling {n} images x {len(selected)} models (1000 denoising steps)")
        with ThreadPoolExecutor(len(selected)) as pool:
            images = dict(zip(selected, pool.map(lambda m: generators[m](n, seed), selected)))
        progress(0.8, desc="labelling")
        rows, galleries = {}, {}
        for name in selected:
            labels = labeller(images[name])
            calls = [counted(l, p, threshold) for l, p in labels]
            rows[name] = bias_metrics(calls)
            items = [(im, f"{c.replace('_', ' ')} ({p:.2f})" if c != UNCOUNTED else f"not counted: {l} ({p:.2f})", c)
                     for im, (l, p), c in zip(images[name], labels, calls)]
            # Low-density classes first, so the rare images are easy to find.
            galleries[name] = [(im, caption) for im, caption, c in sorted(items, key=lambda i: i[2] not in GROUPS["low"])]
        updates = []
        for name in MODELS:
            if name in galleries:
                updates.append(gr.update(value=galleries[name], visible=True, label=f"{name}: {MODELS[name][3]}"))
            else:
                updates.append(gr.update(value=None, visible=False))
        note = (f"**{n} images per model, seed {seed}, threshold {threshold}.** An unbiased model would make about "
                f"4.8% of its counted images low-density (D0's share); the table gives the expected count per model. "
                "Small runs are noisy, so the last column gives each model's "
                "log2 DOR on its 4,750 stored images.")
        live = [row + [stored_rows[name]["log2 DOR"]] for row, name in zip(table(rows), rows)]
        return [note, live, share_chart(rows, f"Live: {n} images per model, seed {seed}"), *updates]

    def browse(name, cls):
        hits = [(path, f"{label.replace('_', ' ')} ({p:.2f})") for path, label, p in stored_labels[name]
                if counted(label, p, 0.9) == cls]
        return hits[:120], f"**{name}** drew **{len(hits)}** confident {cls.replace('_', ' ')} images out of {EVAL_IMAGES} " \
                            f"(about {EVAL_IMAGES * D0_SHARES[cls]:.0f} if it kept its D0 share)."

    def audit(files, threshold):
        if not files:
            raise gr.Error("Upload some images first.")
        images = [Image.open(f).convert("RGB").resize((64, 64), Image.BICUBIC) for f in files]
        labels = labeller(images)
        calls = [counted(l, p, threshold) for l, p in labels]
        row = bias_metrics(calls)
        gallery = [(im, f"{c.replace('_', ' ')} ({p:.2f})" if c != UNCOUNTED else f"not counted: {l} ({p:.2f})")
                   for im, (l, p), c in zip(images, labels, calls)]
        return table({"uploaded images": row}), gallery

    with gr.Blocks(title="KiloBytes: density bias in recursive generative models") as demo:
        gr.Markdown("# Higher-density bias under synthetic-data mixing\n"
                    "IE 643 course project, team KiloBytes: Neha Yadav (23b1807), Tanish Sahu (23b1816). "
                    "Does training on generated data make a model favour frequent classes, and do our mitigations stop it?")
        with gr.Tab("Live generation"):
            with gr.Row():
                selected = gr.CheckboxGroup(list(MODELS), value=DEFAULT_LIVE, label="Models (all start from the same noise)")
            with gr.Row():
                n = gr.Slider(16, 500, value=100, step=1, label="Images per model (about 60 s per 100)")
                seed = gr.Number(SAMPLE_SEED, precision=0, label="Seed")
                threshold = gr.Slider(0.5, 0.99, value=0.9, step=0.01, label="Labeller confidence to count an image")
                run = gr.Button("Generate and measure", variant="primary")
            note = gr.Markdown()
            live_table = gr.Dataframe(headers=LIVE_COLUMNS, label="Class counts and bias metrics (vs D0)", wrap=True)
            live_chart = gr.Plot(label="Class shares")
            galleries = [gr.Gallery(visible=False, columns=10, height=420, object_fit="contain") for _ in MODELS]
            run.click(generate, [selected, n, seed, threshold], [note, live_table, live_chart, *galleries])
        with gr.Tab("Full evaluation (4,750 images)"):
            gr.Markdown("Stored results: 4,750 images per model from the same seeds, labeller v2 at p ≥ 0.9. "
                        f"Without bias each low class would get about {EVAL_IMAGES * D0_SHARES['human']:.0f} images.")
            gr.Dataframe(value=table(stored_rows), headers=TABLE_COLUMNS, wrap=True)
            gr.Plot(value=share_chart(stored_rows, "Stored evaluation, 4,750 images per model"))
            with gr.Row():
                browse_model = gr.Dropdown(list(MODELS), value="G0 (baseline)", label="Model")
                browse_class = gr.Dropdown(CLASSES, value="butterfly", label="Class")
                browse_button = gr.Button("Show images")
            browse_note = gr.Markdown()
            browse_gallery = gr.Gallery(columns=12, height=480, object_fit="contain")
            browse_button.click(browse, [browse_model, browse_class], [browse_gallery, browse_note])
        with gr.Tab("Audit images"):
            gr.Markdown("Upload any images (real test images or a generator's output). Each is resized to 64×64, "
                        "labelled by labeller v2, and the class mix is compared with D0.")
            with gr.Row():
                files = gr.File(file_count="multiple", file_types=["image"], label="Images")
                audit_threshold = gr.Slider(0.5, 0.99, value=0.9, step=0.01, label="Confidence to count an image")
            audit_button = gr.Button("Label and measure", variant="primary")
            audit_table = gr.Dataframe(headers=TABLE_COLUMNS, wrap=True)
            audit_gallery = gr.Gallery(columns=10, height=420, object_fit="contain")
            audit_button.click(audit, [files, audit_threshold], [audit_table, audit_gallery])
        with gr.Tab("Pipeline"):
            gr.Markdown(PIPELINE_MD)
            workflow = ROOT / "report/figures/workflow.png"
            if workflow.is_file():
                gr.Image(str(workflow), show_label=False)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
                        allowed_paths=[str(ROOT / "results"), str(ROOT / "report/figures")])


if __name__ == "__main__":
    main()
