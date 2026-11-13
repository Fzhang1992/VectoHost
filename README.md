# VirusHost3Class

**Three-class viral host-category prediction from nucleotide sequences**

[中文说明](README_CN.md) · [Installation and model setup](docs/INSTALL.md) · [Methods](docs/METHODS.md) · [Reproducibility](docs/REPRODUCIBILITY.md)

VirusHost3Class is a research pipeline for classifying viral sequences as **Insect**, **Plant**, or **Other**. Starting from three FASTA collections, it constructs four sequence-length datasets, computes nucleotide k-mer and pretrained-language-model features, evaluates four classifiers using five-fold cross-validation, and fits separate classifiers for use on new sequences.

| Sequence lengths | Feature representations | Classifiers |
|---|---|---|
| 1,000 bp, 3,000 bp, 5,000 bp, full length | 1-mer, 2-mer, 3-mer, 4-mer, ESM-2, LucaVirus | RF, KNN, SVM, GaussianNB |

**96 combinations** are evaluated independently (4 lengths × 6 representations × 4 classifiers). ESM-2 and LucaVirus embeddings are computed **from the supplied sequences**, not imported from earlier training runs.

## Installation

Linux and Conda are recommended. LucaVirus inference requires the official [LucaVirusTasks](https://github.com/LucaOne/LucaVirusTasks) implementation and corresponding pretrained weights.

```bash
conda env create -f environment.yml
conda activate virushost3class
cp config.example.env config.env
bash scripts/setup_models.sh
```

See [model setup](docs/INSTALL.md) for upstream model and checkpoint instructions.

## Quick example

An executable demonstration is included. It uses **synthetic sequences**, a 3-mer representation, and GaussianNB; no pretrained weights are required.

```bash
python scripts/example_run.py
```

Example outputs are written under `work/example/` and include an OOF summary, a saved classifier, and predictions for two query FASTA records. See [example input](exampleInput/) and [output format](exampleOutput/README.md).

## Reproduce the research workflow

Place one representative full-length sequence per Virus TaxID in three separate files:

```text
input/full_length/Insect_full_length.fasta
input/full_length/Plant_full_length.fasta
input/full_length/Other_full_length.fasta    # .fasta.gz is also supported
```

Each FASTA header must start with a unique numeric virus TaxID, e.g. `>123456|virus=example`. The input collections contain the already selected and concatenated (when segmented) viral genomes. See [input requirements](docs/DATA.md).

Edit `config.env` to point to the FASTAs and to specify the slicing mode and seed. Then run:

```bash
bash run_all.sh --check
mkdir -p logs
nohup bash run_all.sh --run > logs/reproduce_all.log 2>&1 &
tail -f logs/reproduce_all.log
```

The run creates:

```text
work/sequences/              Length-specific FASTAs and TaxID tables
work/features/               Newly extracted feature matrices + provenance
work/results_parts/          Independent model results (resumable)
work/results/                Merged five-fold OOF predictions
work/final/                  96-row summary, confusion matrices, reference comparison
work/models/                 Freshly fitted LucaVirus + SVM classifiers
```

Re-running `bash run_all.sh --run` resumes a run with unchanged inputs/configuration: validated completed steps are skipped. Incomplete ESM-2 batches and LucaVirus vector batches can resume. The pipeline stops rather than silently accepting changed inputs or incomplete results.

## Predict new sequences

After fitting the models with the full workflow:

```bash
python predict.py \
  --input queries.fasta \
  --model-dir work/models/full_length/LucaVirus_SVM \
  --length full_length \
  --lucavirus-tasks external/LucaVirusTasks \
  --output predictions.tsv
```

The output includes the query identifier, predicted class, and probabilities for Insect, Plant, and Other. Fixed-length models reuse the saved training slice seed when available; otherwise specify `--slice-seed` explicitly. Length routing is deliberately explicit. See [training and prediction](docs/TRAIN_AND_PREDICT.md).

## Reproducibility

The full biological FASTAs, pretrained foundation-model weights, and fitted classification models are not distributed with this source package. Reproducing the manuscript's exact values additionally depends on original sequence-window selection and model provenance. The project includes [data and model provenance checks](docs/REPRODUCIBILITY.md) and reports differences rather than replacing observed results with reference values.

## License and acknowledgements

Source code in this repository is offered under the [MIT License](LICENSE). External pretrained models and third-party dependencies retain their own licenses. Please cite the original ESM-2 and LucaVirus publications when using their representations.
