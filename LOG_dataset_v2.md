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

## G0 v2 training (in progress)

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
direct GPU-to-GPU transfers, the mixed-GPU run hung at its first collective. The
training script is still being finalised and is not on this branch yet.

Next: evaluate G0 v2 (DINOv2 HDAI, coverage and CGI, copy check against the
training set), then decide how to build D1 and train G1.
