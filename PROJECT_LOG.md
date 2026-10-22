# Project log: mitigating higher-density bias in generative models

Last updated: **20 October 2026 (Asia/Kolkata)**.

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
| Large DDPM with EMA and attention | Trained; `models/checkpoint_0550.pt` is present; original output directory recorded as `ddpm_ema_attn_large` |
| Initial imbalanced D0 | Created with 15,200 source images |
| Synthetic addition | 3,800 images from the large DDPM, organized into requested class folders |
| Current `dataset-0` | Merged dataset: 19,000 images; `gen-dataset-0` was deleted after merging |
| Compact unconditional G0 | 500-epoch checkpoint is present as `g0.pt` (159 MiB; archive contains `checkpoint_0500`); PyTorch metadata loading was unavailable in the local interpreter |
| D1 for G1 | Partial `dataset-1/` exists: 14,250 images; all 14,250 manifest rows are D0 samples, with no G0-generated images yet |
| G1 training code | Implemented in `train-g1.py`; configured for unconditional training on completed D1; no G1 run recorded |
| Unconditional image-pool generation | Implemented in `generate_density_samples.py` |
| G0 bias evaluation | User-reported 1,000-sample result: HDAI 0.9221, TVD 0.1397, JSD 0.0262 bits; evaluation artifacts are not present in this workspace |
| Recursive comparisons and mitigation results | Planned; no G1/G2 comparison or mitigation result recorded yet |

The old large DDPM is **class conditional**. G0 is **unconditional**. This is an
intentional design change; the old checkpoint cannot be resumed directly as G0.
Training or results on the remote GPU node must be added here when available.

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
- Current group totals are high = 15,000, medium = 3,000, low = 1,000. Their
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

### Classifier inference issue to resolve

`classifier.py` supplies `classify_image` and `classify_images`, but currently
normalizes using mean/std `(0.5, 0.5, 0.5)`, which differs from classifier
training. Align inference preprocessing with the ImageNet normalization above
before using this helper with `models/checkpoint_best.pt` to report bias metrics.
This documentation update does not change that code.

## 4. Trained large DDPM: `ddpm_ema_attn_large`

Architecture/training code: [train.py](train.py). Local checkpoint:
`models/checkpoint_0550.pt`. The checkpoint records original output directory
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

## 5. Compact unconditional G0: current configuration

Code: [train-g0.py](train-g0.py). The 500-epoch checkpoint is available as
`g0.pt`; its PyTorch metadata could not be loaded in the local interpreter.
The workspace also contains `ddpm_g0/checkpoint_0400.pt`. The run settings
below describe current script defaults unless otherwise marked.

| Parameter | Current default / behavior |
|---|---|
| Dataset | Full current `dataset-0`: **19,000 images**, including 3,800 generated additions |
| Conditioning | **Unconditional**; folder labels are ignored; no class embedding instantiated |
| Image size | 64×64 RGB |
| Base channels / timestep embedding dimension | **32** / 256 |
| Trainable parameters | **10,408,483**, derived from the current architecture |
| Epochs / batch size / workers | **50 / 32 / 8** |
| Learning rate | **1e-4**; earlier script version used 2e-4 |
| Optimizer | AdamW, weight decay 0.01 by library default; betas (0.9, 0.999) |
| Loss / diffusion steps | Noise-prediction MSE / 1,000 |
| Beta schedule | Linear 1e-4 → 0.02 |
| EMA decay | **0.999** |
| Attention | Four-head self-attention in encoder stage 3, bottleneck, decoder stage 3 |
| Gradient clipping | Norm 1.0 |
| Precision | CUDA autocast + GradScaler; regular precision on CPU/MPS |
| Image transform | Resize, random horizontal flip, normalize with mean/std 0.5 |
| Data loading | Shuffled; `drop_last=False`, so every image is included each epoch |
| Device | Automatically CUDA, then MPS, then CPU; CLI override available |
| Seed | 42 |
| Preview | **Every epoch**, 16 EMA samples in a 4×4 grid by default |
| Checkpoint frequency | Every 5 epochs and final epoch |
| Output directory | `ddpm_g0` |

EMA is updated after optimizer steps that are not skipped by GradScaler. At each
epoch end the script temporarily loads EMA weights, samples without labels,
saves `samples_0001.png`, `samples_0002.png`, etc., then restores training weights.
`--sample-count` changes the grid's image count. Checkpoints contain raw/EMA
weights, optimizer/scaler states, architecture arguments, epoch, and class-folder
metadata. `dataset.json` records input class counts; `loss.csv`/`loss.png` record
image-weighted mean training loss. Labels in metadata do not condition the model.

The GPU node is reported by the user to support training batches of 512. This is
hardware capacity information, not a saved G0 training setting. Use a CLI
override to train or generate at 512; no such run has been verified locally.

### Size comparison

| Model | Parameters | FP32 weights only | Full training checkpoint |
|---|---:|---:|---:|
| Classifier | 11,173,962 | ≈42.6 MiB, excluding buffers | 134,225,226 bytes ≈128.0 MiB, measured |
| Large DDPM checkpoint 0550 | 60,098,387 | ≈229.3 MiB | 961,892,721 bytes ≈917.3 MiB, measured |
| Compact G0 | 10,408,483 | ≈39.7 MiB | `g0.pt` 159 MiB; archive identifies checkpoint 0500 |

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

## 7. Metrics and matrices

### Already implemented and recorded

- Classifier: sample-weighted cross-entropy loss and accuracy for train,
  validation, and test; best checkpoint selected by validation accuracy.
- Diffusion training: noise-prediction MSE, loss curve, and qualitative sample
  grids. Denoising loss alone does not measure image quality or class bias.
- Dataset: source/generated counts and class/group proportions above.

### Proposed evaluation definitions — not yet implemented or measured

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
| Generation-to-generation change | Compare `q` between G0, G1, G2, and against the original source-mixture reference |
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

## 8. Commands to use

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
python train-g0.py --resume ddpm_g0/checkpoint_0050.pt --epochs 100 --batch-size 512 --device cuda
```

Generate an unfiltered pool using the EMA weights of that G0 checkpoint:

```bash
python generate_density_samples.py --checkpoint ddpm_g0/checkpoint_0050.pt --num-samples 3800 --batch-size 512 --device cuda --output-dir generated_g0
```

Train an unconditional G1 from the completed D1. This starts from fresh
weights; the script only permits resuming checkpoints marked as G1:

```bash
python train-g1.py --data-dir dataset-1 --output-dir ddpm_g1 --epochs 50 --batch-size 32 --device auto
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

## 9. Remaining work and experiment record

1. Repair source-image provenance and recover dog-review records; preserve a
   frozen inventory of the current 19,000-image mixture.
2. Correct classifier inference normalization, then evaluate per-class errors
   and save a confusion matrix before using its predictions for bias analysis.
3. Load `g0.pt` in the training environment and record its checkpoint
   metadata, actual command, hardware, software versions, dataset inventory,
   loss history, and preview examples.
4. Preserve the full evaluation artifacts for the reported 1,000-sample G0
   result and its unfiltered class distribution.
5. Build G1 with replacement fraction `r=0.25`: sample 75% of D0 without
   replacement, generate 25% as many G0 images, classify each generated image
   once for folder bookkeeping, and retain every generated sample. D1 has the
   same total size as D0; its class counts may change with G0's distribution.
   The existing `dataset-1/` contains the 14,250-image D0 sample; run the builder
   to append the 4,750 generated images and complete it.
6. Select and implement a mitigation. Class balancing or classifier-based data
   selection are possible experiments; no method has been chosen or evaluated.
7. Compare baseline and mitigation over repeated seeds; report image quality
   alongside class distribution, then prepare the interface and stage-wise report.

The project currently calls the compact model G0 even though its input dataset
already contains synthetic additions. Preserve that naming history in reports;
do not describe current G0 as trained exclusively on original source images.

For each future run, add a row and link its outputs:

| Run/date | Dataset inventory and synthetic fraction | Model/checkpoint | Command and seed | Measured metrics | Artifacts / notes |
|---|---|---|---|---|---|
| User-reported, 20 Oct 2026 | `dataset-0`, 19,000 images, 20% added synthetic | Unconditional compact G0; 500-epoch archive `g0.pt` | Command, device, and checkpoint metadata not verified | 1,000 generated images: HDAI 0.9221, TVD 0.1397, JSD 0.0262 bits (user-reported) | `g0.pt` present; evaluation output directory currently empty |

## 10. Environment and reproducibility notes

`requirements.txt` lists PyTorch, torchvision, scientific Python libraries,
diffusion packages, and evaluation libraries, but versions are not pinned.
Presence in requirements does not establish that a package was installed or a
metric was run. Save the environment versions from the GPU node with each run.
PyTorch was unavailable in the local Python interpreter during earlier work;
checkpoint settings/results in this log were read from their stored metadata
without running training or inference.

For checkpoint settings and metrics, prefer saved metadata over conversational
estimates or script defaults. For current image counts, prefer the actual folder
inventory over the stale source manifest. The README, scripts, and retained
generation records are the remaining sources for this log.
