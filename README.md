
# PlantCLR: Leveraging Self-Supervised Contrastive Learning for Generalizable Plant Disease Detection

[![Paper: IEEE Access](https://img.shields.io/badge/Paper-IEEE%20Access-blue.svg)](YOUR_PAPER_LINK_HERE)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the **PlantCLR** framework. This repository provides a scalable, self-supervised pipeline for agricultural AI, focusing on feature extraction from unlabelled data to improve diagnostic accuracy in low-resource settings.

---

## 🔬 Research Overview & Methodology

[cite_start]The **PlantCLR** framework addresses two critical bottlenecks in plant pathology: the high cost of expert data annotation and the poor generalization of models from lab-controlled environments to noisy, real-world fields[cite: 405, 406].

### 🛠 The Two-Stage Learning Pipeline
[cite_start]Our methodology decouples representation learning from classification to ensure the model captures intrinsic biological features rather than background noise[cite: 497, 532].

1. [cite_start]**Self-Supervised Pretraining (SimCLR-style):** - Uses a **ConvNeXt-Tiny** backbone as a feature encoder ($f_e$)[cite: 563].
   - [cite_start]Employs a stochastic augmentation strategy ($T$) to generate positive pairs from the same image[cite: 544, 558].
   - [cite_start]**Loss Function:** We utilize the **Normalized Temperature-scaled Cross-Entropy (NT-Xent)** loss[cite: 110].
   
   $$\mathcal{L}_{i,j} = -\log \frac{\exp(s_{i,j}/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(s_{i,k}/\tau)}$$
   [cite_start]*where $s_{i,j}$ is the cosine similarity between projected embeddings $z_i$ and $z_j$[cite: 581, 586].*

2. **Supervised Fine-Tuning:**
   - [cite_start]The projection head is discarded, and a lightweight linear classifier ($h_{\psi}$) is attached to the frozen or unfrozen encoder[cite: 592, 594].
   - [cite_start]Fine-tuned on labeled subsets using **Cross-Entropy Loss with Label Smoothing** to prevent overconfidence and improve robustness[cite: 601, 603].


---
