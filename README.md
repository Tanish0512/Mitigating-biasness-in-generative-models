# Mitigating Higher-Density Bias in Generative Models

## Project Overview

This project investigates whether repeatedly mixing synthetic images into diffusion-model training amplifies a model's preference for higher-density object classes, and develops methods to measure and mitigate that bias.

## Project Description

The team will create a controlled dataset split in which some object classes have approximately equal, relatively high sample counts, while other classes have fewer or very few samples. The study will use real and synthetic/generated image datasets that contain overlapping object categories and suitable class-distribution differences. Class definitions, sample counts, and dataset sources should be documented.

The team will train a relatively small diffusion model on a mixture of real-world and synthetic images. After generating images with the trained model, the team will combine generated images with images from the preceding training level and train the next model. This recursive process may be repeated for up to two additional generations.

At each generation level, the team will:

1. Quantify the model's bias toward higher-density classes using one or more suitable metrics.
2. Analyze how different amounts of synthetic-data mixing affect that bias, including class-wise distribution changes.
3. Apply a proposed mitigation method and evaluate whether it reduces higher-density bias.

The analysis should make clear how bias changes across generations and how much recursive synthetic-data mixing contributes to those changes.

## Expected Deliverables

1. **Mixed real and synthetic dataset** — A controlled dataset containing real and synthetic samples from higher- and lower-density object classes, with class-wise sample distributions clearly documented.
2. **Multi-level recursive diffusion pipeline** — A diffusion model trained on the mixed dataset, followed by image generation and up to two additional training generations that mix generated samples with samples from the previous level.
3. **Higher-density bias metric** — A quantitative metric, or set of metrics, that measures preference for higher-density classes and tracks how it changes at each generation level.
4. **Synthetic-mixing bias analysis** — An analysis of how varying the amount of synthetic data at each recursive level affects higher-density bias, supported by class-wise distribution results.
5. **Bias mitigation method** — A mechanism to reduce higher-density bias during recursive training, evaluated with the proposed metric across all generation levels.

## Final Submission Requirements

The final submission is expected to include:

1. **Trained or fine-tuned models**, along with the project code, documentation, usage instructions, and dependency information.
2. **An interface** through which a user can provide or upload relevant project-specific inputs and test the developed mechanism or methodology. Consult the teaching assistants if the interface requirements are unclear.
3. **Performance metrics** that quantify the effectiveness of the proposed method. If the project has visual outputs, include qualitative results as well.
4. **An extensive, stage-wise report** explaining the rationale for the project steps and documenting major experiments, datasets, dataset extraction and preparation, and the methodology used to build the interface, if one is developed. Further report instructions will be provided later.
5. **Project completion by the intensive assessment**, including all required work described in the project brief.
