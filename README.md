# Mitigating Higher-Density Bias in Generative Models (branch `g0-retrain`)

IE 643 course project, team **KiloBytes**: Neha Yadav (23B1807) and Tanish Sahu (23B1816).

When a generative model is retrained on data that contains its own samples, it can drift towards the
classes that were already frequent and lose the rare ones. This branch contains the final pipeline
**D0 -> G0 -> D1 -> G1**, the bias measurement, two baseline mitigations and the novelty work
(feedback-calibrated class weights, classifier-guided quota sampling and data accumulation), plus a
Gradio test interface. The first version of the pipeline, built before the prep-presentation review,
is kept at the end of this file.

## Pipeline

| Stage | What | Details |
|---|---|---|
| Source pool | 100k real 64x64 images, 10 classes | ImageNet, Cars196, FGVC Aircraft, Happywhale (Kaggle) |
| **D0** | 18,900 images, 20:1 | aircraft / car / ship 6,000 each, butterfly / marine animal / human 300 each; 20% synthetic in every class (60M class-conditional DDPM) |
| **G0** | unconditional DDPM | 22.4M-parameter U-Net, cosine schedule, 1,000 steps, 150k steps, batch 128, AdamW 2e-4, EMA 0.9999 |
| **D1** | D0 mixed with G0 samples | replace 25% or 50% (unfiltered), curated (confident samples in D0 ratios) or accumulated (all of D0 + curated samples) |
| **G1** | same recipe as G0 | trained on each D1 |
| Measurement | labeller v2 | logistic regression on frozen DINOv2 + CLIP features with a background class (99.98% on unseen real images); count only p >= 0.9 |

Metrics: log2 density odds ratio (0 = D0's balance, > 0 favours high-density classes), per-class
amplification A_c = q_c / p_c, HDAI, TVD, DINOv2 coverage, copy check, and 95% bootstrap intervals.

## Main results (4,750 samples per model, same noise seeds)

| Model | log2 DOR [95% CI] | butterfly A | not counted | TVD | coverage hi / lo |
|---|---|---|---|---|---|
| G0 baseline | +0.61 [+0.38, +0.89] | 0.15 | 9.6% | 0.067 | 0.33 / 0.26 |
| G1, 25% mixing | +0.97 [+0.70, +1.29] | 0.18 | 13.0% | 0.044 | 0.26 / 0.22 |
| G1, 50% mixing | +0.85 [+0.60, +1.13] | 0.06 | 14.1% | 0.084 | 0.25 / 0.21 |
| G0 with `n^-1/2` loss weights | -1.50 [-1.63, -1.37] | 0.73 | 14.9% | 0.109 | 0.23 / 0.52 |
| G1 curated + `n^-1/2` | -1.70 [-1.82, -1.57] | 0.48 | 16.6% | 0.114 | 0.20 / 0.46 |
| G1 curated only | +0.50 [+0.27, +0.76] | 0.06 | 9.0% | 0.109 | 0.25 / 0.21 |
| **G0 feedback weights (novelty)** | **+0.09 [-0.10, +0.32]** | 0.64 | 13.1% | **0.018** | 0.29 / 0.31 |
| G1 accumulated + guided (novelty) | +1.04 [+0.78, +1.35] | 0.10 | **8.6%** | 0.081 | 0.28 / 0.21 |
| **G1 full pipeline (novelty)** | **+0.30 [+0.09, +0.52]** | **0.86** | 10.3% | 0.036 | 0.27 / **0.33** |

* Recursive mixing amplifies the bias and the rarest class collapses; HDAI (max 1.05 here) hides it.
* Fixed `n^-1/2` weights bring the rare classes back but over-correct; curation alone does not save them.
* The bias is learnt in training: even perfectly balanced, accumulated or guided data leave it in place.
* **Feedback-calibrated weights** (`w_c <- w_c * A_c^-0.5`, measured on the previous generation)
  remove the G0 bias and, on the accumulated guided D1, restore butterfly from 0.10 to 0.86 of its share
  without over-correcting.

## Repository structure

```
.
├── README.md                  this file
├── requirements.txt
├── docs/
│   ├── README_dataset_v2.md   dataset construction, every script, commands to reproduce all runs
│   ├── LOG_dataset_v2.md      step-by-step log of the rebuilt pipeline
│   └── PROJECT_LOG.md         log of the first version
├── src/                       code of the final pipeline (run from the repository root)
│   ├── data/                  build_dataset0_v2.py, build_dataset1_v2.py (--curate, --accumulate),
│   │                          generate_synthetic.py, diagnose_quality.py
│   ├── generator/             train_diffusion.py (U-Net DDPM, --class-weighting), sample_diffusion.py
│   ├── evaluation/            labeller v2 (train_labeller.py, apply_labeller.py, compare_labellers.py),
│   │                          metrics (evaluate_v2.py, evaluate_features.py, evaluate_clip.py,
│   │                          threshold_metrics.py, novelty_checks.py)
│   ├── novelty/               make_feedback_weights.py, train_noisy_classifier.py, sample_guided.py
│   └── interface/             demo_app.py (Gradio test interface)
├── results/                   labels, configs, loss curves, grids and summaries of every model
│   ├── g0_v2/                 G0
│   ├── g1_v2_r25/             G1, 25% mixing
│   ├── g1_v2_r50/             G1, 50% mixing
│   ├── mitigation/            weighted-loss and curated-mixing models
│   └── novelty/               feedback weights, guided sampling, accumulated D1, bootstrap checks, figures
└── first_version/             scripts of the first 10-class pipeline (see the last section)
```

Scripts are run from the repository root, for example
`python src/generator/train_diffusion.py --data-dir ../data/dataset-0-v2 --output-dir ../runs/g0_v2`;
datasets (`../data`), checkpoints (`../runs`) and full sample folders (`../results`) live next to the
repository and are not in git.

## Test interface

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID python src/interface/demo_app.py --gpus 5,1 --labeller-gpu 0
# open http://<server>:7860, or tunnel: ssh -L 7860:localhost:7860 <user>@<server>
```

Tabs: **Live generation** (pick models, image count and seed; every model starts from the same noise;
class counts, bias metrics and galleries), **Full evaluation** (stored 4,750-image results of all eight
models and a per-class image browser) and **Pipeline** (datasets and metric definitions).

## First version of the pipeline (before the prep-presentation review)

The sections below document the original 10-class pipeline on the `main` branch, kept for reference.
Its scripts are in `first_version/`.

See [PROJECT_LOG.md](docs/PROJECT_LOG.md) for model parameters, recorded metrics,
dataset history, current class counts, commands, and planned experiments.
The current merged `dataset-0` contains **19,000 images**; the 15,200-image
figures in the preparation history below describe the source subset before merging.

### Project Overview

This project investigates whether mixing synthetic images into diffusion-model training amplifies a model's preference for higher-density object classes, and develops methods to measure and mitigate that bias. The agreed study ends at **G1**, covering **G0 and G1** with one recursive training step.

### Project Description

The team will create a controlled dataset split in which some object classes have approximately equal, relatively high sample counts, while other classes have fewer or very few samples. The study will use real and synthetic/generated image datasets that contain overlapping object categories and suitable class-distribution differences. Class definitions, sample counts, and dataset sources should be documented.

The team trains a compact unconditional diffusion model G0 on the mixed dataset D0, then uses G0-generated images together with a sample of D0 to form D1 and train G1. The pipeline is **D0 → G0 → D1 → G1** and ends there.

The completed baseline contains:

- **D0:** 19,000 images, including 3,800 initial synthetic additions.
- **G0:** trained through epoch 500; saved as `models/g0.pt`.
- **D1:** 14,250 D0 images plus 4,750 unfiltered G0 images (`r=0.25`), for 19,000 images total. Generated images are classified once for folder placement and all are retained.
- **G1:** trained on D1 through epoch 700; saved as `models/g1.pt`.

Evaluation and mitigation comparisons remain focused on G0 and G1. Their actual saved training settings differ; see the project log before attributing output changes solely to synthetic mixing.

At both levels, G0 and G1, the team will:

1. Quantify the model's bias toward higher-density classes using one or more suitable metrics.
2. Analyze how different amounts of synthetic-data mixing affect that bias, including class-wise distribution changes.
3. Apply a proposed mitigation method and evaluate whether it reduces higher-density bias.

The analysis should show how bias changes from G0 to G1 and investigate how synthetic-data mixing contributes to that change. Baseline and mitigated comparisons stay within these two levels.

### Expected Deliverables

1. **Mixed real and synthetic dataset** — A controlled dataset containing real and synthetic samples from higher- and lower-density object classes, with class-wise sample distributions clearly documented.
2. **Two-level recursive diffusion pipeline** — G0 trained on D0, followed by generation, D1 construction, and G1 training. The recursive pipeline ends at G1.
3. **Higher-density bias metric** — A quantitative metric, or set of metrics, that measures preference for higher-density classes and tracks how it changes at each generation level.
4. **Synthetic-mixing bias analysis** — An analysis of how varying the amount of synthetic data at each recursive level affects higher-density bias, supported by class-wise distribution results.
5. **Bias mitigation method** — A mechanism to reduce higher-density bias, evaluated against the baseline at G0 and G1 using the proposed metrics.

### Final Submission Requirements

The final submission is expected to include:

1. **Trained or fine-tuned models**, along with the project code, documentation, usage instructions, and dependency information.
2. **An interface** through which a user can provide or upload relevant project-specific inputs and test the developed mechanism or methodology. Consult the teaching assistants if the interface requirements are unclear.
3. **Performance metrics** that quantify the effectiveness of the proposed method. If the project has visual outputs, include qualitative results as well.
4. **An extensive, stage-wise report** explaining the rationale for the project steps and documenting major experiments, datasets, dataset extraction and preparation, and the methodology used to build the interface, if one is developed. Further report instructions will be provided later.
5. **Project completion by the intensive assessment**, including all required work described in the project brief.


### Generation-zero training

`dataset-0/dog` was visually reviewed at its original 64×64 resolution. Of 800
images, 166 were moved outside the training dataset to `dataset-0-excluded/dog`:
images with people or body parts, possible human presence, and four images with
other animals. 634 dog images remained after that review.
Multiple dogs and ordinary backgrounds are allowed. The low resolution limits
certainty, so ambiguous human cases were conservatively excluded. 166 additional dog images were then randomly selected from `images/dog`, visually
reviewed for human presence, and copied in without duplicates. The final dog
count is **800** (15,200 images across all ten classes). Other class folders are
unchanged. The shuffle seed is `20261015`; `dataset-0-excluded/additions.csv`
records source paths, destination paths, and file hashes. All 800 dog images have
unique decoded RGB hashes, with no close matches under the documented perceptual
hash check.

`dataset-0-excluded/manifest.csv` records each original path, excluded path, and
reason; `dataset-0-excluded/review.json` contains the counts. To restore an image,
move its `excluded_path` back to its `original_path` from the manifest.

Train the compact unconditional attention DDPM with EMA. The class folders
provide the image mixture, while class labels are ignored by the model:

```bash
python first_version/train-g0.py --epochs 50
```

Defaults: `dataset-0`, 64×64 RGB, 32 base channels (versus 64 in `train.py`),
batch size 32, 1,000 diffusion steps, EMA decay 0.999, and output `ddpm_g0`.
As an unconditional model this has 10,408,483 parameters versus 38,963,779 in `train.py`
(73.3% fewer).
The narrower model is a starting capacity for this dataset; select training
length using sample quality and evaluation metrics. The imbalanced image
frequencies are retained in the unconditional training mixture for the
density-bias experiment. CUDA, Apple MPS, and CPU are
selected automatically; `--device` overrides the choice.

An EMA sample grid is saved after every epoch. Checkpoints are saved every five
epochs and at the final epoch.
The output also contains `loss.csv`, `loss.png`, and `dataset.json` with the
training class distribution. Checkpoints include raw weights, EMA weights,
optimizer/scaler states, architecture arguments, and class order.

Resume using the same architecture, diffusion settings, and class folders:

```bash
python first_version/train-g0.py --resume ddpm_g0/checkpoint_0050.pt --epochs 100
```

For a class-conditional compact model with the same training defaults, use:

```bash
python train-g0-con.py --data-dir dataset-0 --output-dir ddpm_g0_conditional \
  --epochs 50 --batch-size 32 --device auto
```

This model passes ImageFolder labels into its class embedding and saves the
class order in each checkpoint. It cannot resume an unconditional G0 checkpoint.

For a new G1 experiment, train on the completed `dataset-1/` using the same
compact architecture as G0. The command below is an example, not the saved
700-epoch run’s configuration:

```bash
python first_version/train-g1.py --epochs 50 --batch-size 32 --device auto
```

The default output is `ddpm_g1/`. G1 starts with fresh weights; resuming is
allowed only from a checkpoint marked as G1.

Generate an unlabeled pool from an unconditional G0 checkpoint using:

```bash
python first_version/generate_density_samples.py --checkpoint ddpm_g0/checkpoint_0050.pt --output-dir generated_g0 --batch-size 512
```

Evaluate an unconditional G0 checkpoint by generating 1,000 images, classifying
them with the trained ResNet, and saving class/density bias metrics:

```bash
python evaluate_g0.py --g0-checkpoint ddpm_g0/checkpoint_0050.pt \
  --classifier-checkpoint models/checkpoint_best.pt --data-dir dataset-0
```

The default output is `evaluation_g0/`: generated images, per-image classifier
predictions, per-class reference/generated proportions and amplification, and
`summary.json` with HDAI (high-density group amplification), group amplification,
total variation distance, Jensen–Shannon divergence, and generated-class entropy.
The reference distribution comes from the class counts under `dataset-0`.

D1 was built for the final training level with `r=0.25`: sample 75% of D0
without replacement, generate exactly 25% as many images from unconditional G0,
classify every generated image once, and add all of them to D1. The classifier
predictions only determine `ImageFolder` placement; there is no confidence
filter or class balancing, and G1 remains unconditional:

```bash
python first_version/build_dataset1.py --g0-checkpoint models/g0.pt \
  --classifier-checkpoint models/checkpoint_best.pt --r 0.25
```

The output is `dataset-1/`, with a provenance `manifest.csv` and `summary.json`.
Its total size matches D0, while per-class counts may change according to the
generated G0 distribution. Use the unfiltered G0 evaluation separately to
measure that distribution; D1 includes the full generated batch. If `dataset-1/`
already contains exactly the 75% D0 sample recorded in its manifest, the builder
preserves it and appends the generated batch.

The default pool contains 3,800 images, the total of the proposed 1,000/200/50
per-class targets. Since G0 is unconditional, those class counts cannot be
requested from the sampler; classify the generated pool separately to measure
or select class-wise samples.
