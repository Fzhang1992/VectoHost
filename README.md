# VirusHost3Class

Training and evaluation code for predicting viral host groups: **Insect**, **Plant**, and **Other**. This project supports insect virome studies by distinguishing host categories among viruses detected in insect sequencing data.

## Scripts

| Script | Function |
| --- | --- |
| [train.py](train.py) | Validate inputs, train and evaluate three-class host classifiers, and export evaluation metrics, out-of-fold predictions, and confusion matrices. |

The script uses precomputed 3-mer, ESM2, and LucaVirus features to compare Random Forest, KNN, SVM, and Gaussian Naive Bayes classifiers with five-fold cross-validation.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

Supported datasets: `1000bp`, `3000bp`, `5000bp`, and `full_length`. Replace `/path/to/data` with the directory containing the four length-specific datasets and their prepared inputs.

### Check inputs

Validate inputs without training or writing results:

```bash
python train.py --length 1000bp --data-root /path/to/data --check-only
python train.py --length 3000bp --data-root /path/to/data --check-only
python train.py --length 5000bp --data-root /path/to/data --check-only
python train.py --length full_length --data-root /path/to/data --check-only
```

### Train and evaluate

Each output directory must be new or empty.

```bash
python train.py --length 1000bp --data-root /path/to/data --output-dir ./results/1000bp --run
python train.py --length 3000bp --data-root /path/to/data --output-dir ./results/3000bp --run
python train.py --length 5000bp --data-root /path/to/data --output-dir ./results/5000bp --run
python train.py --length full_length --data-root /path/to/data --output-dir ./results/full_length --run
```
