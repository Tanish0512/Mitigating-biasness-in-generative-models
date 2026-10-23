# dataset-0-v2 log

Log for branch `g0-retrain`. Dates are 2026. The usage guide is
[README_dataset_v2.md](README_dataset_v2.md).

## 22 Oct: why G0 was retrained from scratch

The old G0 (10.4M parameters, 75k steps) produced few recognisable animals and
showed no higher-density bias in class shares. Before changing anything, the
old models and data were checked.

### Classifiers disagree on G0's images

On G0's 4,750 unfiltered samples (the G0 images inside D1):

| Labeller | HDAI | TVD | Low-group amplification |
|---|---:|---:|---:|
| ResNet-18 | 0.913 | 0.123 | 1.77 |
| CLIP ViT-L/14, forced to one of 10 classes | 0.860 | 0.146 | 3.71 |
| CLIP with an extra "unrecognisable" option | 0.895 | 0.127 | 3.27 |
| DINOv2 class-balanced probe (97% balanced accuracy on held-out real images) | 0.964 | 0.096 | not computed |

CLIP and the ResNet agreed on only 65% of images. Each dumps unclear images into
one class (the ResNet into "dog"; CLIP into "fish" and "human"). Classifier
error pushes class shares toward uniform and so hides higher-density bias; with
the most reliable labeller, G0's HDAI was 0.96, close to 1.

Coverage told a different story. Coverage is the fraction of real images that
have a generated image within their 5-nearest-neighbour radius in DINOv2
feature space. G0 covered only 17% of the real data (a held-out real set covers
87%), used only 1,146 distinct real images as nearest neighbours (real: 3,262),
and covered the low-density classes less than the high-density ones: CGI 0.85,
where 1 means no bias. The bias showed up as lost variety, not as shifted class
counts.

### The old D0 data was not the cause

Every D0 image was checked for sharpness and classifier agreement:

- Real images: 98–100% agreement in every class; 10 flagged as wrong class.
- Synthetic images: sharp (0.85–1.13× the real images of the same class), but
  60 flagged as wrong class (bird 23, dog 10, aircraft 9, butterfly 7, cat 7,
  car 4). The flagged synthetic birds are failed samples with no recognisable
  bird.
- G0's samples were not blurrier than real images (Laplacian variance); their
  problem was broken shapes. The DINOv2 labeller was unsure (confidence < 0.5)
  for 72–83% of G0's butterfly, cat, dog and flower images, but only 5–7% of
  its aircraft and car images. Rigid objects came out well; deformable animals
  did not.

## 22 Oct: decisions

1. Remove bird, cat, dog and flower; G0 drew them as blobs.
2. Remove the medium-density group; keep high and low only.
3. High: aircraft, car, ship. Low: butterfly, fish, human.
4. 6,000 / 300 images per class (20:1), 18,900 in total, about the old D0's
   size. Fewer images would not train faster (training time depends on steps)
   and would raise memorisation risk, worst for the low classes.
5. Keep 20% synthetic in every class; reuse old D0's synthetic images and
   generate the missing 1,583 with the same large DDPM.
6. Keep the old fish class's mixed content but rename it `marine_animal`.
7. Screen every image with the rule that flagged the 70 bad ones.

## Findings about the source pool (`images/`)

- `images/fish` is not all fish. Its manifest lists about 4,980 sharks, 2,340
  whales, 1,840 dolphins and 819 fallback copies of older fish images. Whales
  and dolphins are mammals, hence the rename.
- `images/ship` is 8,932 images from the Kaggle "Game of Deep Learning" ship
  dataset, plus 1,068 "preserved previous ship tail" images with no recorded
  source.
- Exact duplicates within the pool: car 26 files, fish 22, ship 8, butterfly 2.
  The builder deduplicates by SHA-256.
- Old D0's real images are byte-identical to the same filenames in `images/`
  (1,200 of 1,200 checked).
- Licences and original sources of the other classes are still undocumented.

## Build result

Generated with `checkpoint_0550.pt` (base 80, conditional, EMA weights), 25%
oversampled: ship 1,438, aircraft 262, car 255, fish 13, human 13.

| Class | Rejected from old D0 | Rejected from pool | Rejected new synthetic | Duplicates skipped |
|---|---:|---:|---:|---:|
| aircraft | 10 | 1 | 2 | 0 |
| car | 4 | 0 | 0 | 2 |
| ship | 0 | 0 | 4 | 1 |
| butterfly | 8 | 0 | 0 | 0 |
| marine_animal | 0 | 0 | 0 | 0 |
| human | 0 | 0 | 0 | 0 |

The old-D0 rejections (22 in the six kept classes) match the earlier diagnosis.
Final dataset: 18,900 images; composition in the README.

## G0 v2 training

G0 is trained from scratch on dataset-0-v2. No old checkpoint or setting is
reused. The same settings will be used for G1 and mitigated runs.

| Setting | Value | Old G0 |
|---|---|---|
| Model | Unconditional U-Net, attention at 16×16 and 8×8, base 48: **22.4M parameters** | 10.4M |
| Dropout | 0.1 | none |
| Noise schedule | **Cosine** (Nichol & Dhariwal 2021), 1,000 steps | linear |
| Steps | **150,000** (about 1,000 epochs) | about 75,000 |
| Batch | 128 global | 128 |
| Optimizer | AdamW 2e-4, 5,000 warmup steps, then constant | 1e-4, later 1e-5 set by hand |
| EMA | 0.9999 | 0.999 |
| Precision | bf16 | fp16 |
| Model of record | EMA weights at step 150,000 | varied |

It runs on 4 GPUs (3 RTX A6000 and 1 RTX 3080 Ti) at about 81 ms per step,
about 3.4 hours in total. `NCCL_P2P_DISABLE=1` is required on this server: with
direct GPU-to-GPU transfers, the mixed-GPU run hung at its first collective.
The run finished on 22 Oct in 3.6 hours (about 86 ms per step on average). Loss,
averaged over 2,000 steps: 0.0448 at 10k, 0.0410 at 50k, 0.0399 at 100k, 0.0394
at 150k.

The EMA sample grids before about 30k steps are noise: with decay 0.9999 and no
EMA warmup, the EMA still holds about 37% of the random initial weights at 10k
steps (0.9999^10000). Raw weights at 10k already drew recognisable vehicles. By
150k the initial weights' share is about 3e-7, so `final.pt` is unaffected.

## 22 Oct: G0 v2 evaluation

4,750 samples from `final.pt` (`sample_diffusion.py`, seed 20261023), evaluated
with `evaluate_v2.py`. The reference is G0's own training set, dataset-0-v2,
split by class into 14,151 reference images and a 4,749-image held-out baseline
with the training ratios. "Unseen real" is 4,749 pool images, never used in
training, with the same ratios. DINOv2 probe balanced accuracy on the baseline:
99.5%.

| Set | HDAI | TVD | Off-manifold | Coverage high | Coverage low | CGI |
|---|---:|---:|---:|---:|---:|---:|
| Training baseline | 1.000 | 0.000 | 9.4% | 0.758 | 0.711 | 1.000 |
| Unseen real | 1.000 | 0.001 | 8.9% | 0.731 | 0.713 | 1.040 |
| **G0 v2** | **0.989** | **0.065** | **19.7%** | **0.331** | **0.264** | **0.849** |

Per class (G0 v2):

| Class | Training share | G0 share | Amplification | Coverage (baseline) |
|---|---:|---:|---:|---:|
| aircraft | 0.317 | 0.312 | 0.98 | 0.333 (0.773) |
| car | 0.317 | 0.267 | 0.84 | 0.289 (0.748) |
| ship | 0.317 | 0.362 | 1.14 | 0.372 (0.755) |
| butterfly | 0.016 | 0.011 | 0.70 | **0.133** (0.724) |
| marine_animal | 0.016 | 0.036 | 2.25 | 0.311 (0.680) |
| human | 0.016 | 0.012 | 0.73 | 0.347 (0.729) |

Findings:

- HDAI cannot show much bias in a 20:1 mixture: the high group is already 95.2%
  of the data, so HDAI's maximum is 1.05. Per-class amplification and coverage
  are the informative measures.
- Butterfly and human are under-produced (0.70x, 0.73x). Marine_animal is
  over-produced (2.25x); its probe-labelled samples include dark or blue
  near-empty images, so part of that excess is failed samples landing in the
  class nearest to "plain blue background".
- Coverage is the clearest bias signal. G0 covers 26-37% of each class except
  butterfly, at 13%: butterflies are the low class G0 learned worst. CGI 0.85
  matches the old G0's value.
- Image quality is much better than the old G0. Vehicles and faces are
  recognisable; butterflies are mostly shapeless.
- No copying: 0.1% of G0 samples have DINOv2 similarity above 0.95 to a training
  image, against 2.7% of unseen real images (the pool holds near-duplicates).
  Median nearest similarity is 0.72 for G0 against 0.84 for unseen real. The 16
  closest pairs are same-type aircraft with different liveries and backgrounds.

Outputs: `results/eval_g0_v2/` (summary.json, per-image labels and nearest
training images, class grids, nearest-pair grid).

Next: decide how to build D1 and train G1.

## 23 Oct: labeller comparison and metric choice

`compare_labellers.py` labelled the same 4,750 G0 v2 samples with eight
labellers. All of them score 99.5-100% balanced accuracy on unseen real pool
images and give HDAI 1.000 on them. Output: `results/labellers_g0_v2/`.

| Labeller | Unrecognisable | HDAI | LDAI | log2 DOR | aircraft | car | ship | butterfly | marine | human |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 probe | 0% | 0.989 | 1.23 | -0.31 | 0.98 | 0.84 | 1.14 | 0.70 | 2.25 | 0.73 |
| DINOv2 probe, reject p < 0.5 | 2.1% | 0.998 | 1.04 | -0.06 | 1.00 | 0.84 | 1.15 | 0.56 | 1.84 | 0.72 |
| CLIP probe | 0% | 0.981 | 1.39 | -0.50 | 0.98 | 0.83 | 1.13 | 0.58 | 2.79 | 0.80 |
| CLIP probe, reject | 2.5% | 0.994 | 1.11 | -0.16 | 1.00 | 0.85 | 1.14 | 0.41 | 2.18 | 0.75 |
| CLIP zero-shot | 0% | 0.998 | 1.04 | -0.06 | 0.99 | 0.84 | 1.17 | 0.37 | 1.34 | 1.41 |
| CLIP zero-shot + unrecognisable | 0.8% | 1.003 | 0.95 | +0.08 | 1.00 | 0.84 | 1.17 | 0.36 | 1.34 | 1.15 |
| Old ResNet-18 | 0% | 0.990 | 1.20 | -0.28 | 0.99 | 0.84 | 1.14 | 0.88 | 1.96 | 0.76 |
| Consensus (DINOv2 and CLIP probes agree, both p >= 0.5) | 4.4% | 1.004 | 0.92 | +0.13 | 1.01 | 0.85 | 1.15 | 0.35 | 1.67 | 0.74 |

Findings:

- The labeller is not why HDAI stays near 1. All eight agree on the pattern:
  car 0.84x, ship 1.15x, butterfly strongly under-produced (0.35-0.88x),
  marine_animal over-produced (1.3-2.8x).
- HDAI is blind for two reasons. Its ceiling is 1.05 at a 95.2% high share.
  And group totals cancel: butterfly's deficit and marine_animal's excess offset
  each other inside the low group, as do car's deficit and ship's excess inside
  the high group. LDAI and DOR share the cancellation.
- Hypothesis: marine_animal shares sea backgrounds with ship (6,000 images), so
  it may borrow density from a high class. Butterfly resembles no high class and
  loses the most. If true, bias depends on how close a rare class is to a
  frequent one, not only on its count.
- Vendi diversity, generated / real at equal sample size (DINOv2, consensus
  labels, n <= 40, noisy): ship 0.69, human 0.71, butterfly 0.85, others about 1.

Metrics adopted:

1. Per-class amplification q_c / p_c, with the worst low class as the headline
   (butterfly 0.35-0.88x).
2. Coverage per class and CGI (label-free; G0 v2 CGI 0.85, butterfly 13%).
3. log2 DOR as the group-level replacement for HDAI (unbounded, 0 = no bias),
   reported beside the per-class numbers because it shares the cancellation.
4. Per-class Vendi ratio for within-class diversity loss.
5. Labels: the consensus labeller, with the DINOv2 probe as a second opinion.

## 23 Oct: labeller v2 (`train_labeller.py`)

The marine_animal excess came from the labeller, not from blue cars. The old
probes saw 225 marine images reweighted 21x and had no "no object" class, so
empty blue frames and broken boats fell into marine_animal.

Labeller v2: logistic regression on concatenated DINOv2 + CLIP ViT-L/14
features, trained on 3,000 pool images per class (no reweighting, unseen
evaluation images excluded) plus a `background` class of 3,000 object-free
images (upscaled corner patches and heavily blurred pool images). Checks:
99.98% balanced accuracy on unseen real images, 0.02% of them called
background, 100% on held-out background images. Saved as
`results/labeller_v2/labeller.pkl`.

Generated images are counted only when the labeller is confident. Unseen real
images barely change with the threshold (0.5% fall below 0.9), so the threshold
removes failed generations, not real objects.

| G0 v2, min. confidence | Not counted | HDAI | log2 DOR | aircraft | car | ship | butterfly | marine | human |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 3.9% | 1.001 | +0.03 | 1.01 | 0.84 | 1.16 | 0.32 | 1.89 | 0.73 |
| 0.8 | 7.2% | 1.011 | +0.37 | 1.02 | 0.84 | 1.17 | 0.21 | 1.39 | 0.74 |
| **0.9** | **9.6%** | **1.017** | **+0.61** | **1.04** | **0.85** | **1.17** | **0.15** | **1.13** | **0.72** |

At p >= 0.9 the 77 marine_animal images are all clear sharks, whales, dolphins
and rays (75 expected from the training share). The 103 below 0.9 are mostly
broken boats and wreck-like shapes: failed ship generations. With failures
removed, the expected direction appears: log2 DOR +0.61 (the high group is
favoured 1.5x in odds), butterfly at 0.15x, human at 0.72x.

Adopted: labeller v2 with a 0.9 confidence threshold for all generations.
