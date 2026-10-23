# G1 v2 results

G1 v2 is trained on D1 v2 with the same recipe, GPUs and seed as G0 v2
([../results_g0_v2/README.md](../results_g0_v2/README.md)), so differences come
from the training data. D1 v2 is 14,175 images from dataset-0-v2 (75% of each
class) plus 4,725 unfiltered G0 v2 samples (r = 0.25), 18,900 in total. These
results cover 4,750 samples from G1's final EMA weights, drawn with the same
sampling seeds as G0's. Discussion: [../LOG_dataset_v2.md](../LOG_dataset_v2.md).
The checkpoint (`final.pt`, 90 MB) is not in git.

## Folders

| Path | Contents |
|---|---|
| `g0_vs_g1_bias.png` | Chart: class amplification and per-class coverage, G0 vs G1 |
| `g0_vs_g1_samples.png` | G0 and G1 sample grids from the same noise seeds, side by side |
| `d1_summary.json` | D1 v2 composition per class and source |
| `training/` | `config.json`, `loss.csv`, G0 and G1 loss curves, EMA sample grids at 50k, 100k and 150k steps |
| `classifier_new_p0.6/` ... `classifier_new_p0.9/` | Labeller v2 (same `labeller.pkl` as G0) at p >= 0.6, 0.7, 0.8 and 0.9: per-image labels, counts and metrics (`summary.json`), class grids (every butterfly, marine_animal and human image; the first 225 of the other classes and of the uncounted images) |
| `evaluation/` | `summary_g0_g1_dinov2.json` (DINOv2 probe, coverage, CGI and copy check for G0 and G1 against dataset-0-v2), `labeller_v2_thresholds_G1v2.json` (counts and metrics at 0.6-0.9), per-image labels, nearest training image, class grids, closest pairs |

## G0 vs G1

Images per class out of 4,750, labeller v2 at p >= 0.9 (expected from the
dataset-0-v2 mix: 1,508 per high class, 75 per low class):

| Class | G0 v2 | G1 v2 |
|---|---:|---:|
| aircraft | 1,416 | 1,446 |
| car | 1,153 | 1,224 |
| ship | 1,590 | 1,360 |
| butterfly | 10 | 12 |
| marine_animal | 77 | **31** |
| human | 49 | 60 |
| low group total | 136 | **103** |
| not counted | 455 (9.6%) | **617 (13.0%)** |
| HDAI (maximum 1.05) | 1.017 | 1.024 |
| log2 density odds ratio (0 = no bias) | +0.61 | **+0.97** |

| DINOv2 space | G0 v2 | G1 v2 |
|---|---:|---:|
| Off-manifold | 19.7% | 27.4% |
| Coverage, high / low | 0.331 / 0.264 | 0.259 / 0.218 |
| CGI | 0.85 | 0.90 |
| Copies (similarity > 0.95) | 0.1% | 0.0% |

The high group's odds against the low group grow from 1.53x the training odds
at G0 to 1.96x at G1, and confident low-class images fall by 24%, mostly
marine_animal. CGI rises only because high-class coverage falls further than
low-class coverage.

## Reproduce

```bash
python build_dataset1_v2.py --d0-dir dataset-0-v2 --g0-samples samples_g0_v2 \
    --labels results_g0_v2/classifier_new_p0.9/labels_G0v2.csv --output-dir dataset-1-v2
torchrun --standalone --nproc_per_node 4 train_diffusion.py --data-dir dataset-1-v2 --output-dir runs/g1_v2
python sample_diffusion.py --checkpoint runs/g1_v2/final.pt --output-dir samples_g1_v2 --start 0 --count 4750 --prefix g1v2
python evaluate_v2.py --data-dir dataset-0-v2 --pool-dir images --generated G0v2=samples_g0_v2:*.png \
    --generated G1v2=samples_g1_v2:*.png --output-dir eval_g0_g1_v2
```

Labeller v2 is applied by loading `results_g0_v2/classifier_new_p0.9/labeller.pkl`
and the DINOv2 + CLIP features (see `train_labeller.py`).
