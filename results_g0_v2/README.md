# G0 v2 results

G0 v2 is an unconditional DDPM trained from scratch on dataset-0-v2. These
results cover 4,750 samples from its final EMA weights. Reasons and the full
discussion are in [../LOG_dataset_v2.md](../LOG_dataset_v2.md). The checkpoint
(`final.pt`, 90 MB) and feature caches are not in git.

## Folders

| Folder | Contents |
|---|---|
| `training/` | `config.json` (all settings, class counts), `loss.csv` (every 100 steps), loss plot, EMA sample grids at 50k, 100k and 150k steps |
| `classifier_old/` | Results with the old classifiers. `summary_dinov2_probe.json` (DINOv2 probe, HDAI, coverage, CGI, copy check), `summary_8_labellers.json` (the same samples under 8 labellers), per-image labels, nearest training image per sample, class grids, closest-pair grids |
| `classifier_new_p0.9/` | Results with labeller v2 (`labeller.pkl`), counting an image only when its top probability is at least 0.9. `summary.json` has thresholds 0.5, 0.8 and 0.9 for G0 v2 and unseen real images, plus the labeller checks. Per-image labels and class grids at 0.9; `marine_hi.png` / `marine_lo.png` are marine_animal images above and below 0.9 |

| `classifier_new_p0.8/` | Labeller v2 at a 0.8 threshold: per-image labels, counts and metrics (`summary.json`), class grids (every butterfly, marine_animal and human image; the first 225 of the other classes and of the uncounted images) |

Images per class out of 4,750 (expected from the training mix: 1,508 per high
class, 75 per low class):

| Class | Old (DINOv2 probe) | New, p >= 0.8 | New, p >= 0.9 |
|---|---:|---:|---:|
| aircraft | 1,484 | 1,433 | 1,416 |
| car | 1,267 | 1,178 | 1,153 |
| ship | 1,721 | 1,632 | 1,590 |
| butterfly | 53 | 15 | 10 |
| marine_animal | 170 | 97 | 77 |
| human | 55 | 52 | 49 |
| not counted | 0 | 343 | 455 |

## Headline numbers

Class amplification is generated share / training share (1 = no bias).

| | Old (DINOv2 probe) | New (labeller v2, p >= 0.9) |
|---|---:|---:|
| aircraft | 0.98 | 1.04 |
| car | 0.84 | 0.85 |
| ship | 1.14 | 1.17 |
| butterfly | 0.70 | **0.15** |
| marine_animal | 2.25 | 1.13 |
| human | 0.73 | 0.72 |
| HDAI (maximum 1.05 here) | 0.989 | 1.017 |
| log2 density odds ratio (0 = no bias) | -0.31 | **+0.61** |
| Images not counted | 0% | 9.6% |

Label-free (DINOv2 space, from `classifier_old/summary_dinov2_probe.json`):
coverage 0.33 for high classes and 0.26 for low (held-out training images: 0.76
and 0.71), butterfly coverage 0.13, CGI 0.85. Copying: 0.1% of samples have
similarity above 0.95 to a training image, against 2.7% for unseen real images.

## Reproduce

```bash
torchrun --standalone --nproc_per_node 4 train_diffusion.py --data-dir dataset-0-v2 --output-dir runs/g0_v2
python sample_diffusion.py --checkpoint runs/g0_v2/final.pt --output-dir samples_g0_v2 --start 0 --count 4750 --prefix g0v2
python evaluate_v2.py --data-dir dataset-0-v2 --pool-dir images --generated G0v2=samples_g0_v2:*.png --output-dir eval_g0_v2
python compare_labellers.py --data-dir dataset-0-v2 --pool-dir images --generated G0v2=samples_g0_v2:*.png \
    --resnet-checkpoint checkpoint_best.pt --dino-cache eval_g0_v2/cache --output-dir labellers_g0_v2
python train_labeller.py --data-dir dataset-0-v2 --pool-dir images --generated G0v2=samples_g0_v2:*.png \
    --dino-cache eval_g0_v2/cache --clip-cache labellers_g0_v2/cache --output-dir labeller_v2
```

On mixed GPUs set `NCCL_P2P_DISABLE=1` for training. `sample_diffusion.py` can
be split across GPUs with `--start` / `--count`; images do not depend on the
split.
