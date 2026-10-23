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
[results_g0_v2/README.md](results_g0_v2/README.md)):

| Script | Purpose |
|---|---|
| `train_diffusion.py` | Trains the unconditional DDPM (22.4M parameters, cosine schedule, 150k steps); one recipe for G0, G1 and mitigated runs |
| `sample_diffusion.py` | Generates samples from a `final.pt`, splittable across GPUs |
| `evaluate_v2.py` | DINOv2 probe class shares, HDAI, coverage, CGI and copy check against dataset-0-v2 |
| `compare_labellers.py` | Same samples under 8 labellers; HDAI, LDAI, density odds ratio, Vendi diversity |
| `train_labeller.py` | Labeller v2: DINOv2 + CLIP features, 3,000 pool images per class and a background class |
| `evaluate_features.py`, `evaluate_clip.py` | Feature, CLIP and ResNet helpers used by the scripts above (they also evaluate the old 10-class G0/G1) |
| `build_dataset1_v2.py` | Builds D1 for G1: 75% of dataset-0-v2 (per class) plus 25% unfiltered G0 v2 samples; summary in `results_g1_v2/d1_summary.json` |
| `diagnose_quality.py` | Sharpness and classifier checks on dataset-0 that led to the class choice |

## Reproduce

Inputs: `images/` (the 100k pool, `Dataset/images.zip` on Drive), `dataset-0/`
(`Dataset-0/dataset-0.zip`), the large DDPM `checkpoint_0550.pt` and the ResNet
`checkpoint_best.pt`. Delete the `__MACOSX/` folders and `.DS_Store` files that
the zips contain before running.

```bash
# 1. New synthetic images (about 18 min on three A6000s; split ship across GPUs)
python generate_synthetic.py --checkpoint models/checkpoint_0550.pt --output-dir synthetic_v2/a \
    --prefix synv2a --seed 20261101 --counts ship=575
python generate_synthetic.py --checkpoint models/checkpoint_0550.pt --output-dir synthetic_v2/b \
    --prefix synv2b --seed 20261102 --counts ship=575
python generate_synthetic.py --checkpoint models/checkpoint_0550.pt --output-dir synthetic_v2/c \
    --prefix synv2c --seed 20261103 --counts aircraft=209 car=204 fish=10 human=10

# 2. Build the dataset (about 10 min on one A6000)
python build_dataset0_v2.py --pool-dir images --d0-dir dataset-0 \
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
