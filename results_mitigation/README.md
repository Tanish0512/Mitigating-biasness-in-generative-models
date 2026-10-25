# Bias mitigation: class-weighted loss and curated mixing

Two mitigations, evaluated against the unmitigated G0 / G1 (same recipe, seeds
and sampling seeds; 4,750 samples per model; labeller v2 at p >= 0.9; DINOv2
evaluation against dataset-0-v2). Discussion: [../LOG_dataset_v2.md](../LOG_dataset_v2.md).
Checkpoints are not in git.

- **Class-weighted loss** (`train_diffusion.py --class-weighting inv_sqrt`): each
  image's noise-prediction loss is multiplied by a_c = n_c^-1/2, normalised so the
  dataset-average weight is 1 (high classes x0.858, low classes x3.838).
- **Curated mixing** (`build_dataset1_v2.py --curate`): D1 takes only G0 samples
  the labeller recognises with p >= 0.9, in D0's class ratios, so D1 has exactly
  6,000 / 300 images per class.

| Model | Trained on | Folder |
|---|---|---|
| G0 mitigated | dataset-0-v2, weighted loss (2 GPUs) | `g0_mit/` (labels for all 9,450 samples used for curation) |
| G1 mitigated | curated D1 from G0 mitigated, weighted loss (4 GPUs) | `g1_mit/` |
| G1 curated only | same curated D1, ordinary loss (2 GPUs) | `g1_curated/` |

Each folder: `training/` (config, loss, EMA sample grids), labels, counts and
metrics at p >= 0.6-0.9, class grids, nearest training images.
`six_models_bias.png` and `summary_all6_dinov2.json` compare all six models;
`d1_curated_summary.json` is the curated D1's composition.

| p >= 0.9 | G0 | G0 mit | G1 25% | G1 50% | G1 mit | G1 curated |
|---|---:|---:|---:|---:|---:|---:|
| butterfly / marine / human | 10/77/49 | 47/259/195 | 12/31/60 | 4/68/38 | 30/255/268 | 4/116/28 |
| low total (expected ~225) | 136 | 501 | 103 | 110 | 553 | 148 |
| not counted | 9.6% | 14.9% | 13.0% | 14.1% | 16.6% | 9.0% |
| log2 density odds ratio | +0.61 | -1.50 | +0.97 | +0.85 | -1.70 | +0.50 |
| coverage high / low | 0.33/0.26 | 0.23/0.52 | 0.26/0.22 | 0.25/0.21 | 0.20/0.46 | 0.25/0.21 |
| copies (> 0.95) | 0.1% | 0% | 0% | 0% | 0% | 0% |

- The weighted loss over-corrects: the bias flips to favour the low classes
  (-1.50, -1.70), with much better low-class coverage but worse high classes.
- Curation alone gives the smallest group bias of any G1 (+0.50) and the fewest
  failed images, but butterfly and human still collapse. Its D1 was built from
  G0 mitigated's samples.
- Next: a gentler weight (~n^-0.15) with curated mixing.

## Reproduce

```bash
torchrun --standalone --nproc_per_node 2 train_diffusion.py --data-dir dataset-0-v2 --output-dir runs/g0_v2_mit --class-weighting inv_sqrt
python sample_diffusion.py --checkpoint runs/g0_v2_mit/final.pt --output-dir samples_g0_v2_mit_all --start 0 --count 9450 --prefix g0v2mit
python apply_labeller.py --labeller results_g0_v2/classifier_new_p0.9/labeller.pkl --images samples_g0_v2_mit_all \
    --cache labeller_cache --output labels_G0v2mit.csv
python build_dataset1_v2.py --d0-dir dataset-0-v2 --g0-samples samples_g0_v2_mit_all --labels labels_G0v2mit.csv \
    --output-dir dataset-1-v2-curated --curate --g0-checkpoint runs/g0_v2_mit/final.pt
torchrun --standalone --nproc_per_node 4 train_diffusion.py --data-dir dataset-1-v2-curated --output-dir runs/g1_v2_mit --class-weighting inv_sqrt
torchrun --standalone --nproc_per_node 2 train_diffusion.py --data-dir dataset-1-v2-curated --output-dir runs/g1_v2_curated
```

Metrics on G0 mitigated use its first 4,750 samples (same seeds as G0 v2).
