# Reproducibility Status

This document tracks the work required to turn the PlantCLR research repository into a independently reproducible implementation.

The final published article is currently the authoritative source for the experimental design and reported results:

- PlantVillage: 99.10% accuracy and 99.04% F1-score
- Cassava Leaf Disease: 96.83% accuracy and 96.70% F1-score
- DOI: https://doi.org/10.1038/s41598-026-45684-x

## Current status

| Component | Status | Requirement before marking complete |
|---|---|---|
| Published article and DOI | Verified | Link resolves to the final Scientific Reports article |
| Final headline metrics | Verified | README values match the final article |
| Citation metadata | Added | CITATION.cff matches the published article |
| Repository ignore rules | Added | Datasets, credentials, checkpoints and generated outputs are excluded |
| Python environment | Pending verification | Export exact packages and versions from the environment used for the final experiments |
| Dataset preparation | Pending verification | Document sources, licences, expected directory structure and split construction |
| Self-supervised pretraining entry point | Pending verification | Identify and test the exact script or notebook used for PlantCLR pretraining |
| Fine-tuning entry point | Pending verification | Identify and test the exact target-domain fine-tuning procedure |
| Evaluation entry point | Pending verification | Reproduce the published metrics from saved predictions or a released checkpoint |
| Random seeds and determinism | Pending verification | Record seeds, deterministic settings and split manifests |
| Model checkpoint | Not released | Release separately with file size, checksum, licence and provenance if permitted |
| Automated tests | Not available | Add lightweight tests for data loading, model forward pass and metric calculation |
| Continuous integration | Not available | Run tests and basic style checks on supported Python versions |

## Rules for the reproducibility release

1. Do not commit datasets, credentials, full checkpoints, caches or bulk experiment outputs to Git.
2. Do not publish a command until it has been executed successfully from a clean environment.
3. Pin or lock dependency versions used for the verified release.
4. Record dataset licences and never redistribute third-party data without permission.
5. Use deterministic split manifests or documented split-generation code.
6. Separate results reported in the published article from later experiments.
7. Provide SHA-256 checksums for externally hosted checkpoints and split manifests.
8. Treat Grad-CAM as qualitative model-behaviour evidence, not biological validation.

## Planned verified interface

The final reproducibility release should expose documented commands for:

1. environment creation;
2. dataset validation and preprocessing;
3. contrastive pretraining;
4. target-domain fine-tuning;
5. evaluation from a fixed checkpoint;
6. generation of the principal tables and figures.

Command names and paths will be added only after the corresponding implementation has been verified.
