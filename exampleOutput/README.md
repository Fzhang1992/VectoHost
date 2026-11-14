# Example output

Run `python scripts/example_run.py`. This creates the following files under `work/example/`:

- `feature_model_summary.tsv`: synthetic 3-mer + GaussianNB cross-validation scores.
- `confusion_matrix.tsv`: synthetic confusion matrix.
- `results/oof_predictions.tsv`: synthetic per-sequence out-of-fold probabilities.
- `model/model.joblib`: a newly fitted toy classifier.
- `predictions.tsv`: classification probabilities for the synthetic `exampleInput/query.fasta`.

Two snapshots, `predictions_synthetic.tsv` and `model_summary_synthetic.tsv`, show the output format from this artificial example. The sequences and all values are **synthetic and not estimates of real viral host performance**. The full research workflow produces 96 model combinations, not one.
