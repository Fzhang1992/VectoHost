# Numerical reproducibility

The analysis reports data- and model-dependent metrics. Running code successfully is not, by itself, evidence that all published metrics have been reproduced. The pipeline records input FASTA hashes, emitted sequence hashes, feature dimensions, available model/checkpoint provenance, training settings, and pooled OOF metrics.

## Essential inputs

1. The curated three-class full-length FASTAs, with original TaxID/representative sequence/segment choices and their SHA256 hashes.
2. The exact length-specific sequence windows when comparing against historical manuscript results. Historically, Insect/Plant windows were reused from prior data preparations, while Other windows followed a deterministic SHA256 rule. Generating new windows may produce different classifier scores.
3. A verified upstream LucaVirusTasks source revision and the v1.0 step-3800000 foundation-model weights. ESM-2 uses `esm2_t6_8M_UR50D`.
4. Training versions, five-fold random seed, fold-specific undersampling and the declared 14 cross-host TaxIDs.

## Evaluation

All 96 feature–classifier–length combinations are evaluated with pooled five-fold OOF probabilities. The 14 cross-host TaxIDs are excluded from the final metrics for every combination. The model-fitting stage uses only training-fold data and performs undersampling within each fold.

`work/final/Figure5_reference_comparison.tsv` records the numeric differences between current LucaVirus+SVM results and the manuscript reference values for the four lengths. The report does **not** adjust the model or alter observed results to force a match. Set `STRICT_REFERENCE=1` only when the required original windows, code revision and checkpoints have been verified.

## Model distribution

This package distributes source code and synthetic examples. It does not include user-specific scientific inputs, foundation model weights or trained deployment classifiers. The source-code license does not grant redistribution permission for unrelated third-party weights or datasets.
