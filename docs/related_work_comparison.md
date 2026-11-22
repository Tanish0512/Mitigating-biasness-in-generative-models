# How our mitigation differs from related methods

**Our setting:** D0 already has the intended class mix (20:1), and the unconditional generator distorts it.
The bias is created by *learning*, then amplified by recursive mixing. The methods below either correct a
biased *dataset*, need class conditioning or an unbiased reference set, act only at sampling time, or
fix weights from class sizes alone. **Feedback-calibrated weights** instead *measure* each generation's
per-class output error and correct the next generation's loss, class by class, without labels at
generation time and without a reference dataset.

| Method | Setting | Mechanism | Needs | Difference from ours |
|---|---|---|---|---|
| Class-balanced loss (Cui et al., CVPR 2019) | classification, long-tailed data | static loss weights from class sizes (effective number of samples) | class counts | Our inverse (n^-1/2) and calibrated (n^-alpha) weights belong to this static family. Feedback weights come from the generator's *measured* amplification A_c, so classes of equal size get different weights (butterfly 2.55, marine 0.92 in G0) |
| CBDM (Qin et al., CVPR 2023) | class-*conditional* diffusion, long-tailed data | regulariser that transfers knowledge to tail-class conditionals | labels as conditioning input | Our generator is unconditional; labels only scale the loss. No change to the architecture or sampler |
| TIW-DSM (Kim et al., ICLR 2024) | unconditional diffusion, *biased dataset* | time-dependent importance weights from a density ratio between biased and unbiased data | a small unbiased reference set and a discriminator | Our data is not biased; the *model's output* is. We need no reference set or discriminator, only the D0 class shares and a labeller, and the weights are updated per generation |
| Minority guidance (Um et al., ICLR 2024) | unconditional diffusion | sampling-time guidance towards low-density samples (minority score) | no retraining | Acts only at sampling time on one model. Ours changes training so the bias does not compound across generations; our classifier guidance (B) only supplies rare samples for curation |
| Algorithmic reparation (Wyllie, Shumailov, Papernot, FAccT 2024) | recursive training, fairness feedback loops | curate representative training data / batches each generation | group labels | Closest to our curated mixing. We show curation alone fails for a diffusion model (balanced curated D1 still gives butterfly 0.06x) because the bias is learnt; feedback on the loss is needed |
| Accumulation (Gerstgrasser et al., 2024) | recursive training, model collapse | keep all real data and add synthetic data | none | Used as our component A; alone it does not remove the class bias (G1 A+B: log2 DOR +1.04) |
| Classifier guidance (Dhariwal & Nichol, NeurIPS 2021) | diffusion sampling | gradient of a noise-aware classifier | a classifier p(y given x_t, t) | Used as our component B, only to fill curation quotas for classes the model rarely draws |

## Evidence behind the "learnt, not inherited" claim

* The curated D1 has exactly D0's class ratios, yet the unweighted G1 trained on it draws butterflies at 0.06x.
* Per-class denoising error on held-out real D0 images (`results/novelty/denoising_loss/`): inverse weighting
  lowers the low-class error by 17-23% and raises the high-class error by only 2-4%, mostly at mid noise
  (t = 200-600) where object shape forms. Feedback weights change only the under-learnt class (butterfly -13%,
  every other class within +/-1%).
