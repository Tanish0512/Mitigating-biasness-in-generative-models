# Project log: mitigating higher-density bias in generative models

Last updated: **22 October 2026 (Asia/Kolkata)**.

This is the working record of datasets, trained models, parameters, metrics,
decisions, and planned experiments. Update it whenever a dataset, training
configuration, checkpoint, or evaluation changes. Values marked **checkpoint**
come from saved checkpoint metadata; **current default** means a setting in the
source code, not evidence that a training run used that setting.

## 1. Objective and present status

Investigate whether recursive synthetic-data mixing amplifies preference for
classes that occur more frequently in the training data. Measure changes in
class distribution, compare mixing amounts, and evaluate a mitigation method.

| Component | Status and evidence |
|---|---|
| Ten-class source dataset | `images/`: 100,000 PNGs, 10,000 per class |
| Classification model | Trained ResNet-18; `models/checkpoint_best.pt` is present |
| Large DDPM with EMA and attention | Trained through epoch 550; current checkpoint `models/ddpm.pt` |
| Initial imbalanced D0 | Created with 15,200 source images |
| Synthetic addition | 3,800 images from the large DDPM, organized into requested class folders |
| Current `dataset-0` | Merged dataset: 19,000 images; `gen-dataset-0` was deleted after merging |
| Compact unconditional G0 | Trained through epoch 500; `models/g0.pt`, saved `epoch=499` |
| D1 for G1 | Complete: 14,250 D0 images + 4,750 unfiltered G0 images = 19,000; summary, manifest, and folder inventory agree |
| Compact unconditional G1 | Training completed through epoch 700 on D1; `models/g1.pt`, saved `epoch=699`, generation tag `G1` |
| Unconditional image-pool generation | Implemented in `generate_density_samples.py` |
| G0 bias evaluation | User-reported 1,000-sample result: HDAI 0.9221, TVD 0.1397, JSD 0.0262 bits; evaluation artifacts are not present in this workspace |
| G0–G1 comparison and mitigation results | Pending; no completed comparison or mitigation result recorded yet |

The old large DDPM is **class conditional**. G0 is **unconditional**. This is an
intentional design change; the old checkpoint cannot be resumed directly as G0.
The agreed scope, confirmed on **22 October 2026**, ends at **G1**:
**D0 → G0 → D1 → G1**, with one recursive mixing/training step.
Baseline training is complete. Remaining experiments cover evaluation and
mitigation comparisons at G0 and G1, followed by the interface and report.

## 2. Dataset history and current class counts

`ImageFolder` uses this alphabetical class order. Preserve it when interpreting
classifier logits or conditional checkpoint labels.

| Label | Class | Density group | `images/` | Original D0 | Synthetic addition | Current `dataset-0` |
|---:|---|---|---:|---:|---:|---:|
| 0 | aircraft | High | 10,000 | 4,000 | 1,000 | 5,000 |
| 1 | bird | High | 10,000 | 4,000 | 1,000 | 5,000 |
| 2 | butterfly | Medium | 10,000 | 800 | 200 | 1,000 |
| 3 | car | High | 10,000 | 4,000 | 1,000 | 5,000 |
| 4 | cat | Medium | 10,000 | 800 | 200 | 1,000 |
| 5 | dog | Medium | 10,000 | 800 | 200 | 1,000 |
| 6 | fish | Low | 10,000 | 200 | 50 | 250 |
| 7 | flower | Low | 10,000 | 200 | 50 | 250 |
| 8 | human | Low | 10,000 | 200 | 50 | 250 |
| 9 | ship | Low | 10,000 | 200 | 50 | 250 |
| | **Total** | | **100,000** | **15,200** | **3,800** | **19,000** |

- The selected per-class density ratio is **20:4:1**. The earlier suggestion of
  2,000/500/100 was replaced by the user's choice of 4,000/800/200.
- The merge preserved that ratio: each class received another 25% of its source
  count. Synthetic images are **20% of the merged dataset**: 3,800 / 19,000.
- D0 group totals are high = 15,000, medium = 3,000, low = 1,000. Their
  probability masses are 78.95%, 15.79%, and 5.26%, respectively.
- The correct addition total is **3,800**; the earlier conversational total of
  4,700 was an arithmetic error.
- `images/` was preserved. Dataset source identity, licensing, and whether all
  original source images are real photographs still need explicit documentation.

### Sampling, review, and provenance

The original D0 was sampled without replacement from sorted PNG paths using
Python `Random(0)`. Source class directories were copied into `dataset-0` at the
selected counts.

The README records a later dog-image review: 166 images were excluded because of
human presence, ambiguous human presence, or other animals; 166 replacements
were selected with shuffle seed `20261015`, bringing dog back to 800 source
images before merging. It also records an image-hash review. These are historical
records: the referenced `dataset-0-excluded/` review files are **not present
locally** at this update, so those review claims have not been independently
rechecked for this log.

| Record | Meaning and current limitation |
|---|---|
| `dataset-0/manifest.csv` | Original 15,200 source selections; 166 listed files are now missing, consistent with the recorded dog replacements. It is not an up-to-date inventory of the whole merged dataset. |
| `dataset-0/generated_manifest.csv` | 3,800 generated selections; all listed paths currently exist |
| `dataset-0/generated_summary.json` | Generation source `models/checkpoint_0550.pt`, seed `20261015`, 64×64 images, requested class counts |

During merging, generated images were moved into corresponding class folders;
there were no filename collisions. The generated manifest and summary were
retained under the names above, and `gen-dataset-0` was removed.

Generated class folder names record **requested conditioning labels**, not
independently verified semantic classes. No classifier-verified distribution of
these 3,800 images has been recorded. Repair the source manifest and recover the
review artifacts before claiming complete image-level provenance.

## 3. Classification model

Training code: [train_resnet.py](train_resnet.py). Checkpoint:
`models/checkpoint_best.pt`.

### Architecture and saved training configuration

| Item | Value | Evidence |
|---|---|---|
| Network | ResNet-18 with 10 outputs | Source and checkpoint tensor shapes |
| Input | 64×64 RGB | Checkpoint |
| Stem | 3×3 convolution, stride 1, padding 1; max-pool replaced by identity | Source |
| Initialization | ImageNet pretrained; central 3×3 crop of pretrained stem weights | Saved `pretrained=True` and source |
| Trainable parameters | **11,173,962** | Checkpoint shapes, excluding BatchNorm buffers |
| Dataset | `images/` | Checkpoint |
| Split | Stratified 80% train, 10% validation, 10% test; seed 42 | Checkpoint and source |
| Split sizes on current source dataset | 80,000 / 10,000 / 10,000; per class 8,000 / 1,000 / 1,000 | Derived from counts and split code |
| Batch size / workers | 128 / 8 | Checkpoint |
| Requested final epoch / saved best epoch | 100 / **74** | Checkpoint; does not establish the final epoch actually completed |
| Optimizer | AdamW; betas (0.9, 0.999), epsilon 1e-8 | Saved optimizer |
| Configured learning rate / weight decay | 2e-5 / 1e-4 | Checkpoint |
| Learning rate saved in optimizer | 1.7289686274214123e-5 | Saved optimizer |
| Scheduler | CosineAnnealingLR; T_max=100, eta_min=0 | Saved scheduler |
| Loss | Cross-entropy with label smoothing 0.1 | Current source |
| Device / seed | CUDA / 42 | Checkpoint |
| Resume source | `resnet_runs/checkpoint_last.pt` | Checkpoint; path not present locally |

Current script defaults differ from this saved run: 20 epochs, learning rate
3e-4, and 4 workers. Saved configuration describes a resumed run; settings and
results for its earlier stages are not fully available here. The saved history
contains 24 rows, epochs 51–74. Its scheduler records `last_epoch=24`; do not
assume an uninterrupted 100-epoch cosine schedule from the available metadata.

Training augmentation: resize, random horizontal flip, and color jitter with
brightness/contrast/saturation = 0.15. Validation and test use resize only.
All three use ImageNet normalization: mean `(0.485, 0.456, 0.406)` and standard
deviation `(0.229, 0.224, 0.225)`.

### Recorded classifier results

| Metric | Value | Source |
|---|---:|---|
| Training accuracy at best epoch | 100.00% | Epoch-74 saved history |
| Training loss at best epoch | 0.500391 | Epoch-74 saved history |
| Validation accuracy | **98.84%** | Best checkpoint |
| Validation loss | 0.531897 | Epoch-74 saved history |
| Test accuracy | **98.80%** | Best checkpoint |
| Test loss | 0.533079 | Best checkpoint |

Loss values use the smoothed-label criterion, so they should not be compared
directly with unsmoothed cross-entropy loss. Confusion matrices, per-class
precision/recall/F1, and generated-image classification accuracy are not stored
in the available checkpoint.

### Classifier preprocessing

`classifier.py` supplies `classify_image` and `classify_images`, but currently
normalizes using mean/std `(0.5, 0.5, 0.5)`, which differs from classifier
training. Align inference preprocessing with the ImageNet normalization above
before using this helper with `models/checkpoint_best.pt` to report bias metrics.
`build_dataset1.py` already uses the correct ImageNet normalization directly,
as also recorded in D1’s summary. The earlier G0 evaluation was reported by
the user; its evaluation script and result directory are currently absent.

## 4. Trained large DDPM: `ddpm_ema_attn_large`

Architecture/training code: [train.py](train.py). Current local checkpoint:
`models/ddpm.pt` (archive `checkpoint_0550`; historical name
`models/checkpoint_0550.pt`). The checkpoint records original output directory
`ddpm_ema_attn_large` and dataset `images/`.

| Parameter | Saved checkpoint value |
|---|---|
| Conditioning | **Class conditional**, 10-class embedding; `unconditional=False` |
| Image size | 64×64 RGB |
| Base channels / timestep embedding dimension | **80** / 256 |
| Trainable parameters | **60,098,387** |
| Batch size / workers | 128 / 8 |
| Training epoch | 550 completed; stored zero-based `epoch=549` |
| Diffusion steps | 1,000 |
| Beta schedule | Linear from 1e-4 to 0.02, per current implementation |
| Objective | MSE between predicted noise and sampled Gaussian noise |
| Optimizer | AdamW; betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01 |
| Learning rate at save | **1e-6** |
| EMA decay | **0.9999** |
| Gradient clipping | Norm 1.0, per current implementation |
| CUDA precision | Autocast with GradScaler, per current implementation |
| Checkpoint interval | Every 5 epochs and final epoch |
| Preview samples | 16; grid after every epoch, using EMA |
| Seed | 42 |
| Resume source | `ddpm_ema_attn_large/checkpoint_0470.pt` |

Current `train.py` defaults are 64 base channels, 50 epochs, batch size 64,
learning rate 2e-4, and 4 workers. Those defaults do **not** describe the saved
80-channel model. The full earlier learning-rate/training history is not
available; the checkpoint's 1e-6 is its final saved setting.

The U-Net has residual convolution blocks, GroupNorm with 8 groups, SiLU,
sinusoidal timestep embeddings, and four-head self-attention. Attention operates
in the third encoder stage, bottleneck, and matching decoder stage. At 64×64
input, those attention resolutions are 16×16 and 8×8. Training images are resized,
horizontally flipped, and normalized to `[-1, 1]`.

DDPM outputs include `loss.csv`, `loss.png`, sample grids, and checkpoints. No
DDPM loss history, FID, KID, or measured density-bias results were available
locally for this checkpoint when this log was created.

## 5. Trained compact unconditional G0 and G1

### Saved training configurations

G0 is stored at `models/g0.pt`; G1 is stored at `models/g1.pt`. These values
were read from the saved checkpoint metadata, including tensor shapes, without
loading tensor payloads or running inference. Both models have **10,408,483
parameters**, 32 base channels, 256-dimensional timestep embeddings, and no
class embedding. Both train unconditionally on 64×64 RGB images.

| Setting | G0 checkpoint | G1 checkpoint |
|---|---|---|
| Training script | `train-g0.py` | `train-g1.py` |
| Dataset | `dataset-0` | `dataset-1` |
| Completed epoch | **500** (`epoch=499`) | **700** (`epoch=699`) |
| Requested final epoch in saved args | 600; only completion through 500 is established | 700 |
| Original output directory | `ddpm_g0` | `ddpm_g1` |
| Batch size / workers | **128 / 8** | **256 / 8** |
| Learning rate at save | **1e-5** | **5e-5** |
| EMA decay | **0.999** | **0.99** |
| Diffusion steps | 1,000 | 1,000 |
| Optimizer | AdamW, weight decay 0.01 | AdamW, weight decay 0.01 |
| Device / seed | CUDA / 42 | CUDA / 42 |
| Resume source in saved args | `ddpm_g0/checkpoint_0400.pt` | `ddpm_g1/checkpoint_0500.pt` |
| Save interval / preview count | Every 5 epochs / 16 | Every 5 epochs / 16 |
| Generation tag | Not stored | `G1` |

G1’s resume source is its own epoch-500 checkpoint. The saved final checkpoint
does not reconstruct every earlier training command or hyperparameter change.
G0 and G1 differ in completed epochs, batch size, saved learning rate, and EMA
decay. Any later difference in their output distributions cannot be attributed
solely to synthetic-data mixing without controlling these training differences.

The source code uses noise-prediction MSE, a linear beta schedule from 1e-4 to
0.02, random horizontal flips, normalization to [-1, 1], gradient clipping at
1.0, and CUDA autocast with GradScaler. Current CLI defaults may differ from
the actual saved settings above.

EMA is updated after optimizer steps that are not skipped by GradScaler. At each
epoch end the script temporarily loads EMA weights, samples without labels,
saves `samples_0001.png`, `samples_0002.png`, etc., then restores training weights.
`--sample-count` changes the grid's image count. Checkpoints contain raw/EMA
weights, optimizer/scaler states, architecture arguments, epoch, and class-folder
metadata. `dataset.json` records input class counts; `loss.csv`/`loss.png` record
image-weighted mean training loss. Labels in metadata do not condition the model.

### Size comparison

| Model | Parameters | FP32 weights only | Full training checkpoint |
|---|---:|---:|---:|
| Classifier | 11,173,962 | ≈42.6 MiB, excluding buffers | 134,225,226 bytes ≈128.0 MiB, measured |
| Large DDPM checkpoint 0550 | 60,098,387 | ≈229.3 MiB | 961,892,721 bytes ≈917.3 MiB, measured |
| Compact G0 | 10,408,483 | ≈39.7 MiB | `models/g0.pt`: 166,852,430 bytes ≈159.1 MiB |
| Compact G1 | 10,408,483 | ≈39.7 MiB | `models/g1.pt`: 166,852,430 bytes ≈159.1 MiB |

G0 has about **82.7% fewer parameters**, or **5.8 times fewer**, than the saved
80-channel large DDPM. Relative to a 64-channel unconditional `train.py` model
(38,963,779 parameters), G0 has about 73.3% fewer parameters. These are different
comparisons. Weight memory estimates exclude activations and training overhead.

## 6. Generation history and current sampling behavior

The original generated addition used the conditional checkpoint 0550 with
requested counts 1,000 per high-density class, 200 per medium-density class,
and 50 per low-density class, for a total of **3,800**. Its saved seed is
`20261015`, and the generation summary is retained in `dataset-0`.

After deciding that G0 must be unconditional, `generate_density_samples.py` was
changed to generate an **unlabeled pool**. Its current defaults are:

| Setting | Value |
|---|---|
| Checkpoint | `ddpm_g0/checkpoint_0050.pt` |
| Output directory | `generated-dataset-0` |
| Total samples | 3,800 |
| Batch size | 16, overridable to 512 |
| Seed | 20261015 |
| Model weights | EMA when available, otherwise raw model weights |
| Class input | None; conditional checkpoints are explicitly rejected |
| Output files | Individual PNGs, `manifest.csv`, `summary.json` |

The 1,000/200/50 figures retained in its summary are historical quota targets,
not guaranteed output counts. To measure unconditional model bias, classify the
complete unfiltered generated pool. Selecting images until quotas are filled
would change the observed distribution and should be logged as a separate
data-selection experiment.

`generate_all_samples.py` handles class-conditional attention checkpoints.
`generate_samples.py` imports the older non-attention architecture from
`train_ddpm.py`; do not use it as the sampler for the current attention models.

### Completed D1 construction

`dataset-1/summary.json` and `dataset-1/manifest.csv` record the completed
build; the current folder inventory contains 19,000 images and matches the
summary’s per-class counts. The manifest contains 14,250 `D0_sample` rows and
4,750 `G0_generated` rows.

With **r = 0.25**, D1 consists of **14,250 sampled D0 images (75%)** and
**4,750 newly generated G0 images (25%)**, using seed **20261020**. Every
newly generated image was classified once and retained. No confidence
threshold, rejection sampling, or class-quota balancing was applied. The
classifier labels determine folders; unconditional G1 ignores these labels.
Only the total size is fixed across D0 and D1; class counts can change.

| Class | D0 total | Retained D0 | New G0 samples | D1 total |
|---|---:|---:|---:|---:|
| aircraft | 5,000 | 3,771 | 1,128 | 4,899 |
| bird | 5,000 | 3,731 | 1,337 | 5,068 |
| butterfly | 1,000 | 757 | 189 | 946 |
| car | 5,000 | 3,735 | 969 | 4,704 |
| cat | 1,000 | 771 | 170 | 941 |
| dog | 1,000 | 746 | 516 | 1,262 |
| fish | 250 | 188 | 117 | 305 |
| flower | 250 | 186 | 167 | 353 |
| human | 250 | 179 | 23 | 202 |
| ship | 250 | 186 | 134 | 320 |
| **Total** | **19,000** | **14,250** | **4,750** | **19,000** |

D1’s high/medium/low folder totals are **14,671 / 3,149 / 1,180**, respectively.
The 25% replacement fraction counts only new G0 samples: retained D0 images
already include earlier synthetic additions, so D1 is not 75% real images.

The summary records generator `models/g0.pt` and classifier
`models/classifier.pt`. The latter path is absent locally; the available
classifier is `models/checkpoint_best.pt`. Their identity has not been verified.
The earlier 14,250-image partial-folder observation is superseded by this
completed build; its retained class counts also differ from that partial copy.

## 7. Metrics and matrices

### Already implemented and recorded

- Classifier: sample-weighted cross-entropy loss and accuracy for train,
  validation, and test; best checkpoint selected by validation accuracy.
- Diffusion training: noise-prediction MSE, loss curve, and qualitative sample
  grids. Denoising loss alone does not measure image quality or class bias.
- Dataset: source/generated counts and class/group proportions above.

### Metric definitions and evaluation status

G0’s reported 1,000-image evaluation gave **HDAI = 0.9221**, **TVD = 0.1397**,
and **JSD = 0.0262 bits**. These are user-reported results; the evaluation files
are not currently available. No G1 generated-image HDAI, TVD, JSD, FID, or KID
result has been recorded. D1’s composition is a training-data inventory, not
an evaluation of G1’s outputs.

HDAI is defined here as high-density group amplification `q_H / p_H`:
1 means matching the reference share, above 1 means overrepresentation, and
below 1 means underrepresentation relative to that reference.

For class `i`, define training proportion `p_i = n_i / sum_j n_j` and generated
proportion `q_i = m_i / M`, where `m_i` is the classifier count in an unfiltered
pool of `M` generated images. Use the correct reference `p` for each run.

| Metric / matrix | Definition and purpose |
|---|---|
| Classifier confusion matrix | `C[i,j]` counts true class `i` predicted as `j`; report counts and row-normalized values on held-out labeled data |
| Per-class precision/recall/F1 and macro-F1 | Check whether high overall classifier accuracy hides unequal class performance |
| Generated class-count table | Report `n_i`, `p_i`, `m_i`, `q_i` for every generation and mixing setting |
| Class amplification | `A_i = q_i / p_i`; above 1 means overrepresentation relative to that training distribution |
| Group amplification | `A_H = sum(q_i for i in high) / sum(p_i for i in high)`; likewise for medium and low |
| Total variation distance | `TVD(p,q) = 0.5 * sum_i abs(p_i - q_i)`; 0 means matching proportions |
| Jensen–Shannon divergence | Let `r=(p+q)/2`; `JSD=0.5*KL(p||r)+0.5*KL(q||r)`, using base-2 logs and zero-term handling |
| Generation-to-generation change | Compare `q` between G0 and G1, and against the common D0 reference |
| FID / KID | Candidate image-quality metrics with fixed feature extractor, preprocessing, reference split, and sample size; no scores available yet |

For every metric, record checkpoint, seed, total generated images, classifier
version, preprocessing, class order, and mixing fraction. If uncertain images
are rejected, report the confidence rule and rejected count; retain the full-pool
distribution as well. Classifier predictions on generated images need visual
checks because accuracy on source test images does not establish accuracy on
generated images.

The existing large DDPM used `images/`, which includes the classifier's test
partition. That partition is held out from classifier training but is not an
independent holdout for the generator. Define a suitable independent reference
before making held-out generator-quality claims.

## 8. Example commands using current artifacts

These are usage examples, not a reconstruction of the completed training runs.
The checkpoint settings in section 5 are the record of those runs.

Install dependencies in the intended environment:

```bash
python -m pip install -r requirements.txt
```

Train a new unconditional G0 on all current `dataset-0` images:

```bash
python train-g0.py --data-dir dataset-0 --output-dir ddpm_g0 --epochs 50 --batch-size 512 --device cuda
```

Resume a compatible unconditional G0 checkpoint to a larger final epoch count:

```bash
python train-g0.py --resume models/g0.pt --output-dir ddpm_g0_continued --epochs 600 --batch-size 128 --learning-rate 1e-5 --device cuda
```

Generate an unfiltered pool using the EMA weights of that G0 checkpoint:

```bash
python generate_density_samples.py --checkpoint models/g0.pt --num-samples 3800 --batch-size 512 --device cuda --output-dir generated_g0
```

Optionally continue the saved G1 beyond its completed epoch 700, preserving
its saved batch size, learning rate, and EMA decay:

```bash
python train-g1.py --data-dir dataset-1 --resume models/g1.pt --output-dir ddpm_g1_continued --epochs 800 --batch-size 256 --learning-rate 5e-5 --ema-decay 0.99 --device cuda
```

The sampler requires an empty or new output directory. Changing sampling batch
size can change the generated images even with the same seed, so record both.

A fresh classifier run with the current training script can be started with:

```bash
python train_resnet.py --data-dir images --output-dir resnet_runs --epochs 100 --batch-size 128 --learning-rate 2e-5 --num-workers 8 --device cuda
```

This command is a new run, not an exact replay of the historical resumed
classifier experiment. Likewise, the saved large DDPM's final arguments alone
cannot reproduce its missing earlier training stages.

## 9. Completed work, remaining work, and experiment record

Completed: source/classifier preparation, the 550-epoch large conditional DDPM,
the mixed D0, unconditional G0 through epoch 500, a reported G0 evaluation,
D1 construction at r=0.25, and unconditional G1 through epoch 700.
`train-g0-con.py` was created earlier as a separate conditional variant, but
is absent from the current workspace; no completed run of it is recorded. The primary G0/G1 pipeline is unconditional.

Next steps:

1. Generate an unfiltered G1 evaluation pool from `models/g1.pt` and classify
   it with the same validated preprocessing. Report HDAI, class/group shares,
   amplification, TVD, and JSD against D1 and a common D0 reference.
2. Preserve evaluation outputs, training curves, sample grids, and full run
   commands. Resolve the historical classifier filename in D1’s summary.
3. Complete classifier per-class error analysis and qualitative review of
   generated images; retain the reported G0 evaluation artifacts.
4. Compare G0/G1 under matched training settings and, where feasible, multiple
   seeds and mixing fractions. The saved runs used different training settings.
5. Implement and evaluate mitigation at G0 and G1 against their unfiltered
   baselines, with matched settings for each comparison. Any mitigated recursive
   branch also ends at G1. No completed mitigation result is recorded yet.
6. Repair historical D0 provenance, add image-quality evaluation, and finish
   the interface and stage-wise report.

| Stage | Dataset | Saved model | Recorded progress / metrics |
|---|---|---|---|
| Large conditional DDPM | `images/` | `models/ddpm.pt` | 550 completed epochs; produced D0 synthetic additions |
| G0 | D0: 19,000 images, including 3,800 initial synthetic additions | `models/g0.pt` | 500 completed epochs; user-reported HDAI 0.9221, TVD 0.1397, JSD 0.0262 bits on 1,000 samples |
| D1 construction | 14,250 D0 samples + 4,750 unfiltered G0 samples | `dataset-1/summary.json`, `manifest.csv` | Complete; 19,000 images; r=0.25; seed 20261020 |
| G1 | Completed D1 | `models/g1.pt` | 700 completed epochs; G1 output evaluation pending |
| Mitigation at G0 and G1 | Comparisons within the two-level study | Not yet recorded | Pending |

## 10. Environment and reproducibility notes

`requirements.txt` lists PyTorch, torchvision, scientific Python libraries,
diffusion packages, and evaluation libraries, but versions are not pinned.
Presence in requirements does not establish that a package was installed or a
metric was run. Save the environment versions from the GPU node with each run.
PyTorch was unavailable in the local Python interpreter during earlier work;
checkpoint metadata was read using a restricted metadata reader that substitutes
tensor shapes for tensor reconstruction. No training or inference was run during
this log update. Saved settings and tensor shapes were inspected; tensor values
and model quality were not validated by that inspection.

For checkpoint settings and metrics, prefer saved metadata over conversational
estimates or script defaults. For current image counts, prefer the actual folder
inventory over the stale source manifest. The README, scripts, and retained
generation records are the remaining sources for this log.
