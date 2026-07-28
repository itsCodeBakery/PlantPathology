# PlantCLR: Contrastive Self-Supervised Pretraining for Generalizable Plant Disease Detection

[![Scientific Reports](https://img.shields.io/badge/Scientific%20Reports-2026-1f6f8b)](https://doi.org/10.1038/s41598-026-45684-x)
[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41598--026--45684--x-blue)](https://doi.org/10.1038/s41598-026-45684-x)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Official repository associated with **PlantCLR**, published in *Scientific Reports* in 2026.

PlantCLR evaluates a contrastive self-supervised pretraining and fine-tuning pipeline for plant-disease classification under cross-dataset transfer with target-domain fine-tuning. The framework combines SimCLR-style contrastive pretraining with a lightweight convolutional classifier to improve representation transfer while maintaining computational efficiency.

> **Paper:** Syed Shayan Ali Shah et al., “PlantCLR: contrastive self-supervised pretraining for generalizable plant disease detection,” *Scientific Reports*, vol. 16, 2026.  
> **DOI:** [10.1038/s41598-026-45684-x](https://doi.org/10.1038/s41598-026-45684-x)

## Research Motivation

Supervised plant-disease classifiers depend heavily on large labeled datasets and may not transfer reliably across datasets collected under different imaging conditions. PlantCLR investigates whether contrastive self-supervised pretraining can learn transferable visual representations before target-domain fine-tuning.

The study evaluates PlantCLR on:

- **PlantVillage**, used for in-domain plant-disease classification.
- **Cassava Leaf Disease**, used to assess transfer with target-domain fine-tuning.

## Method Overview

The experimental pipeline contains two principal stages:

1. **Contrastive self-supervised pretraining**  
   Two augmented views of each image are processed using a SimCLR-style objective to learn visual representations without relying on class labels during pretraining.

2. **Target-domain fine-tuning and evaluation**  
   The learned encoder is adapted for plant-disease classification and evaluated using accuracy, precision, recall, F1-score, confusion matrices, feature-embedding visualization and qualitative explanation maps.

<p align="center">
  <img src="Plots/CLR_Dia.png" alt="PlantCLR contrastive pretraining and classification workflow" width="760">
</p>

## Published Results

The following results reproduce the values reported in the final published article.

| Dataset | Accuracy | F1-score |
|---|---:|---:|
| PlantVillage | **99.10%** | **99.04%** |
| Cassava Leaf Disease | **96.83%** | **96.70%** |

These values supersede metrics from preliminary experiments or earlier manuscript versions.

## Visual Analysis

The published study uses t-SNE to examine class separation in the learned feature space and Grad-CAM to provide qualitative evidence about image regions influencing model predictions.

Grad-CAM maps are interpretability aids. They do not constitute lesion annotations, biological validation or proof of causal model reasoning.

<p align="center">
  <img src="Plots/PL_tSNE.png" alt="PlantCLR t-SNE visualization" width="620">
</p>

<p align="center">
  <img src="Plots/gcPlantVillage%20(1).png" alt="PlantCLR Grad-CAM examples on PlantVillage" width="760">
</p>



## Authors

- **Syed Shayan Ali Shah**
- Faisal Saeed
- Muhammad Umair Raza
- Abdul Rehman
- Muhammad Shaheryar
- Il-Min Kim
- Sangseok Yun
- Jae-Mo Kang

## Citation

If you use this work, please cite the published article:

```bibtex
@article{shah2026plantclr,
  title   = {PlantCLR: contrastive self-supervised pretraining for generalizable plant disease detection},
  author  = {Shah, Syed Shayan Ali and Saeed, Faisal and Raza, Muhammad Umair and Rehman, Abdul and Shaheryar, Muhammad and Kim, Il-Min and Yun, Sangseok and Kang, Jae-Mo},
  journal = {Scientific Reports},
  volume  = {16},
  year    = {2026},
  doi     = {10.1038/s41598-026-45684-x},
  url     = {https://doi.org/10.1038/s41598-026-45684-x}
}
```

## Responsible Use

PlantCLR is a research system for plant-disease image classification. Predictions should not be treated as a substitute for assessment by qualified agricultural or plant-pathology professionals.

## Contact

For questions about the research or repository, use [GitHub Issues](https://github.com/itsCodeBakery/PlantPathology/issues) or contact **Syed Shayan Ali Shah** at [shayan.ali@imsciences.edu.pk](mailto:shayan.ali@imsciences.edu.pk).

## License

The repository code is provided under the [MIT License](LICENSE). The published article and third-party datasets remain subject to their respective licenses and terms of use.
