# G1 v2 with 50% G0 images (r = 0.5)

Same recipe, GPUs and seed as G1 v2 ([../results_g1_v2/README.md](../results_g1_v2/README.md));
only the mixing amount differs. D1-r50 keeps 50% of each dataset-0-v2 class
(9,450 images) and adds 9,450 unfiltered G0 v2 samples. 4,750 samples from the
final EMA weights, same sampling seeds as G0 and G1. Discussion:
[../LOG_dataset_v2.md](../LOG_dataset_v2.md). The checkpoint is not in git.

| Path | Contents |
|---|---|
| `g0_g1_r25_r50_bias.png` | Class amplification and per-class coverage: G0, G1-25%, G1-50% |
| `d1_summary.json` | D1-r50 composition per class and source |
| `training/` | `config.json`, `loss.csv`, EMA sample grids at 50k, 100k, 150k steps |
| `labels_G1v2r50.csv` | Labeller v2 label and probability per image |
| `labeller_v2_thresholds_G1v2r50.json` | Counts and metrics at p >= 0.6-0.9 |
| `class_grids/`, `nearest_*` | Images per class (DINOv2 probe), nearest training image per sample |

Labeller v2, p >= 0.9 (expected 1,508 per high class, 75 per low class):

| Class | G0 v2 | G1 25% | G1 50% |
|---|---:|---:|---:|
| aircraft / car / ship | 1416 / 1153 / 1590 | 1446 / 1224 / 1360 | 1400 / 1039 / 1531 |
| butterfly / marine / human | 10 / 77 / 49 | 12 / 31 / 60 | 4 / 68 / 38 |
| not counted | 9.6% | 13.0% | 14.1% |
| log2 density odds ratio | +0.61 | +0.97 | +0.85 |
| coverage high / low | 0.33 / 0.26 | 0.26 / 0.22 | 0.25 / 0.21 |

## Reproduce

```bash
python build_dataset1_v2.py --d0-dir dataset-0-v2 --g0-samples samples_g0_v2_all \
    --labels labels_g0_v2_all.csv --output-dir dataset-1-v2-r50 --r 0.5
torchrun --standalone --nproc_per_node 4 train_diffusion.py --data-dir dataset-1-v2-r50 --output-dir runs/g1_v2_r50
python sample_diffusion.py --checkpoint runs/g1_v2_r50/final.pt --output-dir samples_g1_v2_r50 --start 0 --count 4750 --prefix g1v2r50
python apply_labeller.py --labeller results_g0_v2/classifier_new_p0.9/labeller.pkl --images samples_g1_v2_r50 \
    --cache labeller_cache --output labels_G1v2r50.csv
python threshold_metrics.py labels_G1v2r50.csv labeller_v2_thresholds_G1v2r50.json
```
