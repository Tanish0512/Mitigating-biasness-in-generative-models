# dataset-0-v2 (branch `g0-retrain`)

This branch prepares a new D0 for training G0 **from scratch**. It does not
change `main`'s code, `README.md` or `PROJECT_LOG.md`. Reasons and results are
in [LOG_dataset_v2.md](LOG_dataset_v2.md).

## What changed from dataset-0

| | dataset-0 (main) | dataset-0-v2 (this branch) |
|---|---|---|
| Classes | 10 | **6**: bird, cat, dog and flower removed |
| Density groups | high / medium / low (20:4:1) | **high / low only (20:1)** |
| High-density classes | aircraft, bird, car: 5,000 each | **aircraft, car, ship: 6,000 each** |
| Low-density classes | fish, flower, human, ship: 250 each | **butterfly, marine_animal, human: 300 each** |
| Synthetic share | 20% (3,800) | 20% in every class (3,780) |
| Total | 19,000 | **18,900** |
| Screening | none | images rejected when both classifiers disagree with the folder |

`marine_animal` is the old `fish` class renamed. The source folder holds sharks,
whales and dolphins (see the log), so "fish" was not an accurate name.

### Per-class composition

| Class | Group | Real from old D0 | Real from `images/` | Synthetic from old D0 | New synthetic | Total |
|---|---|---:|---:|---:|---:|---:|
| aircraft | high | 3,999 | 801 | 991 | 209 | 6,000 |
| car | high | 3,998 | 802 | 996 | 204 | 6,000 |
| ship | high | 200 | 4,600 | 50 | 1,150 | 6,000 |
| butterfly | low | 240 | 0 | 60 | 0 | 300 |
| marine_animal | low | 200 | 40 | 50 | 10 | 300 |
| human | low | 200 | 40 | 50 | 10 | 300 |
| **Total** | | **8,837** | **6,283** | **2,197** | **1,583** | **18,900** |

Every class is 80% real and 20% synthetic. All synthetic images come from the
same large class-conditional DDPM (`checkpoint_0550.pt`) that made D0's
original synthetic additions.

## Scripts

| Script | Purpose |
|---|---|
| `generate_synthetic.py` | Generates per-class images from the large conditional DDPM, with 25% extra to allow for screening |
| `build_dataset0_v2.py` | Selects, screens, deduplicates and copies images into `dataset-0-v2/` and writes `manifest.csv` and `summary.json` |

Both scripts depend only on `train.py` and `train_resnet.py` from `main`, plus
`transformers` (DINOv2) and `scikit-learn`.

G0 v2 training and evaluation (results and commands in
[results/g0_v2/README.md](results/g0_v2/README.md)):

| Script | Purpose |
|---|---|
| `train_diffusion.py` | Trains the unconditional DDPM (22.4M parameters, cosine schedule, 150k steps); one recipe for G0, G1 and mitigated runs; `--class-weighting inv_sqrt` weights each image's loss by n_c^-1/2 |
| `sample_diffusion.py` | Generates samples from a `final.pt`, splittable across GPUs |
| `evaluate_v2.py` | DINOv2 probe class shares, HDAI, coverage, CGI and copy check against dataset-0-v2 |
| `compare_labellers.py` | Same samples under 8 labellers; HDAI, LDAI, density odds ratio, Vendi diversity |
| `train_labeller.py` | Labeller v2: DINOv2 + CLIP features, 3,000 pool images per class and a background class |
| `evaluate_features.py`, `evaluate_clip.py` | Feature, CLIP and ResNet helpers used by the scripts above (they also evaluate the old 10-class G0/G1) |
| `build_dataset1_v2.py` | Builds D1 for G1: (1-r) of each dataset-0-v2 class plus r unfiltered G0 samples (`--r`, default 0.25); `--curate` takes only confident G0 samples in D0's class ratios |
| `apply_labeller.py` | Labels a folder of generated images with labeller v2 |
| `threshold_metrics.py` | Counts and bias metrics at labeller thresholds 0.6-0.9 from an `apply_labeller.py` CSV |
| `diagnose_quality.py` | Sharpness and classifier checks on dataset-0 that led to the class choice |
| `demo_app.py` | Test interface (Gradio): live generation from all eight models with labels and bias metrics against D0 and the stored 4,750-image results with an image browser |
| `make_feedback_weights.py` | Novelty: per-class loss weights from a model's measured class amplification, `w_c <- w_c * A_c^-eta`, clipped and normalised |
| `train_noisy_classifier.py` | Novelty: noise-aware classifier `p(y \| x_t, t)` on D0 for classifier guidance |
| `sample_guided.py` | Novelty: classifier-guided samples of one class from an unchanged generator |
| `novelty_checks.py` | Bootstrap 95% intervals of log2 DOR and per-class amplification, and the confidence filter's per-class pass rate |

### Test interface

Needs `pip install gradio`, the six `runs/*/final.pt`, labeller v2 (`results/labeller_v2/labeller.pkl`)
and the stored sample folders under `results/`.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID python src/interface/demo_app.py --gpus 5,1 --labeller-gpu 0
# open http://<server>:7860, or tunnel: ssh -L 7860:localhost:7860 <user>@<server>
```

Live sampling runs the full 1000 denoising steps: about 80 s for 100 images from three models
on three A6000s. Small live runs are noisy, so the live table also shows each model's log2 DOR
on its 4,750 stored images.

## Reproduce

Inputs: `images/` (the 100k pool, `Dataset/images.zip` on Drive), `dataset-0/`
(`Dataset-0/dataset-0.zip`), the large DDPM `checkpoint_0550.pt` and the ResNet
`checkpoint_best.pt`. Delete the `__MACOSX/` folders and `.DS_Store` files that
the zips contain before running.

```bash
# 1. New synthetic images (about 18 min on three A6000s; split ship across GPUs)
python src/data/generate_synthetic.py --checkpoint models/checkpoint_0550.pt --output-dir synthetic_v2/a \
    --prefix synv2a --seed 20261101 --counts ship=575
python src/data/generate_synthetic.py --checkpoint models/checkpoint_0550.pt --output-dir synthetic_v2/b \
    --prefix synv2b --seed 20261102 --counts ship=575
python src/data/generate_synthetic.py --checkpoint models/checkpoint_0550.pt --output-dir synthetic_v2/c \
    --prefix synv2c --seed 20261103 --counts aircraft=209 car=204 fish=10 human=10

# 2. Build the dataset (about 10 min on one A6000)
python src/data/build_dataset0_v2.py --pool-dir images --d0-dir dataset-0 \
    --synthetic-dirs synthetic_v2/a synthetic_v2/b synthetic_v2/c \
    --resnet-checkpoint models/checkpoint_best.pt --output-dir dataset-0-v2
```

The build seed is 20261022. `dataset-0-v2/manifest.csv` records, for every
image, its class, density group, source (`real_d0`, `real_pool`, `synthetic_d0`,
`synthetic_v2`), the path it was copied from, its original pool source path
(when the pool recorded one) and its SHA-256.

Note: `generate_samples.py` on `main` imports the U-Net from `train_ddpm.py`,
which does **not** match `checkpoint_0550.pt` (that checkpoint was trained with
`train.py`'s attention U-Net, base 80). `generate_synthetic.py` uses `train.py`.

## Novelty assessment

The intensive phase showed that G0 is already biased, that fixed `n^-1/2` loss weights over-correct and
that curated mixing does not save the rarest class. Two checks located the cause: the curated D1 had
exactly D0's class ratios yet its G1 drew 4 butterflies, and the `p >= 0.9` filter keeps 93-96% of
predicted vehicles but only 32% of predicted butterflies. The bias is learnt in training, so we tested:

* **C. Feedback-calibrated class weights**: each generation's loss weights are corrected by the
  previous model's measured amplification, `w_c <- w_c * A_c^-0.5`, clipped to [0.7, 3] and normalised
  to mean 1 (`train_diffusion.py --class-weighting file`).
* **B. Classifier-guided quota sampling**: a noise-aware classifier steers the unchanged baseline G0
  towards butterflies (scale 5; 0.2% -> 18.7% confident) so curation can fill its quota.
* **A. Accumulation**: D1 keeps all of D0 and adds 4,725 curated samples (23,625 images).

All runs use the same recipe on 4x A6000 (32 per GPU, about 4.1 h each). Results on 4,750 samples per
model, labeller v2 at p >= 0.9, 95% bootstrap intervals:

| Model | log2 DOR [95% CI] | butterfly | marine | human | not counted | TVD | coverage hi / lo |
|---|---|---|---|---|---|---|---|
| G0 baseline | +0.61 [+0.38, +0.89] | 0.15 | 1.13 | 0.72 | 9.6% | 0.067 | 0.33 / 0.26 |
| G0 `n^-1/2` weights | -1.50 [-1.63, -1.37] | 0.73 | 4.04 | 3.04 | 14.9% | 0.109 | 0.23 / 0.52 |
| **G0 feedback (C)** | **+0.09 [-0.10, +0.32]** | 0.64 | 1.39 | 0.79 | 13.1% | **0.018** | 0.29 / 0.31 |
| G1 25% mixing | +0.97 [+0.70, +1.29] | 0.18 | 0.47 | 0.92 | 13.0% | 0.044 | 0.26 / 0.22 |
| G1 curated only | +0.50 [+0.27, +0.76] | 0.06 | 1.69 | 0.41 | 9.0% | 0.109 | 0.25 / 0.21 |
| **G1 A+B** | +1.04 [+0.78, +1.35] | 0.10 | 0.89 | 0.51 | **8.6%** | 0.081 | 0.28 / 0.21 |
| **G1 A+B+C** | **+0.30 [+0.09, +0.52]** | **0.86** | 0.55 | **1.05** | 10.3% | 0.036 | 0.27 / **0.33** |

Feedback weights remove the G0 bias without over-correcting; data fixes alone (A+B) reduce failures but
not the bias; adding C on the same D1 restores butterfly from 0.10 to 0.86 of its share. Marine animal
falls to 0.55 in G1 (feedback-loop over-correction). Labels, configs, loss curves, weights, bias checks,
coverage and figures are in `results/novelty/`.

```bash
# C at G0: weights from baseline G0, then train on 4 GPUs
python src/novelty/make_feedback_weights.py --labels results/g0_v2/classifier_new_p0.9/labels_G0v2.csv \
    --data-dir ../data/dataset-0-v2 --output ../runs/feedback/g0_fb_weights.json
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 src/generator/train_diffusion.py --data-dir ../data/dataset-0-v2 \
    --output-dir ../runs/g0_fb --class-weighting file --class-weights-file ../runs/feedback/g0_fb_weights.json

# B: noise-aware classifier and guided butterflies from baseline G0
python src/novelty/train_noisy_classifier.py --data-dir ../data/dataset-0-v2 --output-dir ../runs/noisy_classifier
python src/novelty/sample_guided.py --checkpoint ../runs/g0_v2/final.pt --classifier ../runs/noisy_classifier/final.pt \
    --class-name butterfly --scale 5 --count 1250 --output-dir ../results/guided_g0_v2

# A: accumulated, curated D1 (pool = unguided G0 samples + guided butterflies)
python src/data/build_dataset1_v2.py --d0-dir ../data/dataset-0-v2 --g0-samples ../results/pool_g0_v2_guided \
    --labels ../results/pool_g0_v2_guided_labels.csv --output-dir ../data/dataset-1-accum-guided \
    --r 0.25 --accumulate --curate

# G1 A+B+C: weights updated from G0 feedback's measurement
python src/novelty/make_feedback_weights.py --labels results/novelty/g0_fb/labels_G0fb.csv \
    --data-dir ../data/dataset-1-accum-guided --target-dir ../data/dataset-0-v2 \
    --previous ../runs/feedback/g0_fb_weights.json --output ../runs/feedback/g1_fb_weights.json
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 src/generator/train_diffusion.py --data-dir ../data/dataset-1-accum-guided \
    --output-dir ../runs/g1_accum_guided_fb --class-weighting file --class-weights-file ../runs/feedback/g1_fb_weights.json

# Intervals and filter pass rates
python src/evaluation/novelty_checks.py --data-dir ../data/dataset-0-v2 --output-dir ../results/novelty_checks_all \
    --model G0=results/g0_v2/classifier_new_p0.9/labels_G0v2.csv --model G1_ABC=results/novelty/g1_accum_guided_fb/labels_G1agfb.csv
```

## Novelty part 2: n^-0.15 versus feedback weights (instructor feedback)

Three extra generators, all with the intensive-phase recipe (2 x 64 images per GPU, batch 128, 150k steps):

```bash
# G0 with n^-0.15 weights on D0
torchrun --nproc_per_node 2 src/generator/train_diffusion.py --data-dir ../data/dataset-0-v2 --output-dir ../runs/g0_cal \
    --class-weighting power --class-weight-alpha 0.15
# G1 on the curated D1 with n^-0.15 weights, and with the G1 feedback weights (same data, only the weights differ)
torchrun --nproc_per_node 2 src/generator/train_diffusion.py --data-dir ../data/dataset-1-v2-curated --output-dir ../runs/g1_cur_cal \
    --class-weighting power --class-weight-alpha 0.15
torchrun --nproc_per_node 2 src/generator/train_diffusion.py --data-dir ../data/dataset-1-v2-curated --output-dir ../runs/g1_cur_fb \
    --class-weighting file --class-weights-file ../runs/feedback/g1_fb_weights.json
# intervals for the three models (results/novelty/bias_checks_cal)
python src/evaluation/novelty_checks.py --data-dir ../data/dataset-0-v2 --output-dir ../results/novelty_checks_cal \
    --model G0_cal=results/novelty/g0_cal/labels_g0cal.csv --model G1_cur_cal=results/novelty/g1_cur_cal/labels_g1curcal.csv \
    --model G1_cur_fb=results/novelty/g1_cur_fb/labels_g1curfb.csv
```
Sampling, labelling and the CPU evaluation follow the same commands as the other novelty runs.
