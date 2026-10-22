# Mitigating Higher-Density Bias in Generative Models

See [PROJECT_LOG.md](PROJECT_LOG.md) for model parameters, recorded metrics,
dataset history, current class counts, commands, and planned experiments.
The current merged `dataset-0` contains **19,000 images**; the 15,200-image
figures in the preparation history below describe the source subset before merging.

## Project Overview

This project investigates whether mixing synthetic images into diffusion-model training amplifies a model's preference for higher-density object classes, and develops methods to measure and mitigate that bias. The agreed study ends at **G1**, covering **G0 and G1** with one recursive training step.

## Project Description

The team will create a controlled dataset split in which some object classes have approximately equal, relatively high sample counts, while other classes have fewer or very few samples. The study will use real and synthetic/generated image datasets that contain overlapping object categories and suitable class-distribution differences. Class definitions, sample counts, and dataset sources should be documented.

The team trains a compact unconditional diffusion model G0 on the mixed dataset D0, then uses G0-generated images together with a sample of D0 to form D1 and train G1. The pipeline is **D0 → G0 → D1 → G1** and ends there.

The completed baseline contains:

- **D0:** 19,000 images, including 3,800 initial synthetic additions.
- **G0:** trained through epoch 500; saved as `models/g0.pt`.
- **D1:** 14,250 D0 images plus 4,750 unfiltered G0 images (`r=0.25`), for 19,000 images total. Generated images are classified once for folder placement and all are retained.
- **G1:** trained on D1 through epoch 700; saved as `models/g1.pt`.

Evaluation and mitigation comparisons remain focused on G0 and G1. Their actual saved training settings differ; see the project log before attributing output changes solely to synthetic mixing.

At both levels, G0 and G1, the team will:

1. Quantify the model's bias toward higher-density classes using one or more suitable metrics.
2. Analyze how different amounts of synthetic-data mixing affect that bias, including class-wise distribution changes.
3. Apply a proposed mitigation method and evaluate whether it reduces higher-density bias.

The analysis should show how bias changes from G0 to G1 and investigate how synthetic-data mixing contributes to that change. Baseline and mitigated comparisons stay within these two levels.

## Expected Deliverables

1. **Mixed real and synthetic dataset** — A controlled dataset containing real and synthetic samples from higher- and lower-density object classes, with class-wise sample distributions clearly documented.
2. **Two-level recursive diffusion pipeline** — G0 trained on D0, followed by generation, D1 construction, and G1 training. The recursive pipeline ends at G1.
3. **Higher-density bias metric** — A quantitative metric, or set of metrics, that measures preference for higher-density classes and tracks how it changes at each generation level.
4. **Synthetic-mixing bias analysis** — An analysis of how varying the amount of synthetic data at each recursive level affects higher-density bias, supported by class-wise distribution results.
5. **Bias mitigation method** — A mechanism to reduce higher-density bias, evaluated against the baseline at G0 and G1 using the proposed metrics.

## Final Submission Requirements

The final submission is expected to include:

1. **Trained or fine-tuned models**, along with the project code, documentation, usage instructions, and dependency information.
2. **An interface** through which a user can provide or upload relevant project-specific inputs and test the developed mechanism or methodology. Consult the teaching assistants if the interface requirements are unclear.
3. **Performance metrics** that quantify the effectiveness of the proposed method. If the project has visual outputs, include qualitative results as well.
4. **An extensive, stage-wise report** explaining the rationale for the project steps and documenting major experiments, datasets, dataset extraction and preparation, and the methodology used to build the interface, if one is developed. Further report instructions will be provided later.
5. **Project completion by the intensive assessment**, including all required work described in the project brief.


## Generation-zero training

`dataset-0/dog` was visually reviewed at its original 64×64 resolution. Of 800
images, 166 were moved outside the training dataset to `dataset-0-excluded/dog`:
images with people or body parts, possible human presence, and four images with
other animals. 634 dog images remained after that review.
Multiple dogs and ordinary backgrounds are allowed. The low resolution limits
certainty, so ambiguous human cases were conservatively excluded. 166 additional dog images were then randomly selected from `images/dog`, visually
reviewed for human presence, and copied in without duplicates. The final dog
count is **800** (15,200 images across all ten classes). Other class folders are
unchanged. The shuffle seed is `20261015`; `dataset-0-excluded/additions.csv`
records source paths, destination paths, and file hashes. All 800 dog images have
unique decoded RGB hashes, with no close matches under the documented perceptual
hash check.

`dataset-0-excluded/manifest.csv` records each original path, excluded path, and
reason; `dataset-0-excluded/review.json` contains the counts. To restore an image,
move its `excluded_path` back to its `original_path` from the manifest.

Train the compact unconditional attention DDPM with EMA. The class folders
provide the image mixture, while class labels are ignored by the model:

```bash
python train-g0.py --epochs 50
```

Defaults: `dataset-0`, 64×64 RGB, 32 base channels (versus 64 in `train.py`),
batch size 32, 1,000 diffusion steps, EMA decay 0.999, and output `ddpm_g0`.
As an unconditional model this has 10,408,483 parameters versus 38,963,779 in `train.py`
(73.3% fewer).
The narrower model is a starting capacity for this dataset; select training
length using sample quality and evaluation metrics. The imbalanced image
frequencies are retained in the unconditional training mixture for the
density-bias experiment. CUDA, Apple MPS, and CPU are
selected automatically; `--device` overrides the choice.

An EMA sample grid is saved after every epoch. Checkpoints are saved every five
epochs and at the final epoch.
The output also contains `loss.csv`, `loss.png`, and `dataset.json` with the
training class distribution. Checkpoints include raw weights, EMA weights,
optimizer/scaler states, architecture arguments, and class order.

Resume using the same architecture, diffusion settings, and class folders:

```bash
python train-g0.py --resume ddpm_g0/checkpoint_0050.pt --epochs 100
```

For a class-conditional compact model with the same training defaults, use:

```bash
python train-g0-con.py --data-dir dataset-0 --output-dir ddpm_g0_conditional \
  --epochs 50 --batch-size 32 --device auto
```

This model passes ImageFolder labels into its class embedding and saves the
class order in each checkpoint. It cannot resume an unconditional G0 checkpoint.

For a new G1 experiment, train on the completed `dataset-1/` using the same
compact architecture as G0. The command below is an example, not the saved
700-epoch run’s configuration:

```bash
python train-g1.py --epochs 50 --batch-size 32 --device auto
```

The default output is `ddpm_g1/`. G1 starts with fresh weights; resuming is
allowed only from a checkpoint marked as G1.

Generate an unlabeled pool from an unconditional G0 checkpoint using:

```bash
python generate_density_samples.py --checkpoint ddpm_g0/checkpoint_0050.pt --output-dir generated_g0 --batch-size 512
```

Evaluate an unconditional G0 checkpoint by generating 1,000 images, classifying
them with the trained ResNet, and saving class/density bias metrics:

```bash
python evaluate_g0.py --g0-checkpoint ddpm_g0/checkpoint_0050.pt \
  --classifier-checkpoint models/checkpoint_best.pt --data-dir dataset-0
```

The default output is `evaluation_g0/`: generated images, per-image classifier
predictions, per-class reference/generated proportions and amplification, and
`summary.json` with HDAI (high-density group amplification), group amplification,
total variation distance, Jensen–Shannon divergence, and generated-class entropy.
The reference distribution comes from the class counts under `dataset-0`.

D1 was built for the final training level with `r=0.25`: sample 75% of D0
without replacement, generate exactly 25% as many images from unconditional G0,
classify every generated image once, and add all of them to D1. The classifier
predictions only determine `ImageFolder` placement; there is no confidence
filter or class balancing, and G1 remains unconditional:

```bash
python build_dataset1.py --g0-checkpoint models/g0.pt \
  --classifier-checkpoint models/checkpoint_best.pt --r 0.25
```

The output is `dataset-1/`, with a provenance `manifest.csv` and `summary.json`.
Its total size matches D0, while per-class counts may change according to the
generated G0 distribution. Use the unfiltered G0 evaluation separately to
measure that distribution; D1 includes the full generated batch. If `dataset-1/`
already contains exactly the 75% D0 sample recorded in its manifest, the builder
preserves it and appends the generated batch.

The default pool contains 3,800 images, the total of the proposed 1,000/200/50
per-class targets. Since G0 is unconditional, those class counts cannot be
requested from the sampler; classify the generated pool separately to measure
or select class-wise samples.
