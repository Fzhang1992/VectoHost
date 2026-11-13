# Training and inference

1. Install dependencies with `environment.yml`, configure `config.env`, and obtain the official LucaVirusTasks code and weights ([installation](INSTALL.md)).
2. Run `bash run_all.sh --check` to verify input paths and installed modules.
3. Run `nohup bash run_all.sh --run > logs/reproduce_all.log 2>&1 &`. On interruption, run the same command again after resolving the cause.
4. Inspect `work/final/feature_model_summary.tsv` (96 rows) and `work/final/Figure5_reference_comparison.tsv` (reference differences, if any).

Training is divided into 24 feature/classifier components per sequence length. Completed components are checked against source TaxIDs and merged to form each length's `oof_predictions.tsv`. This prevents the entire 96-model evaluation from restarting after a single interrupted combination.

## New sequence prediction

After the full run finishes, four independently fitted LucaVirus+SVM models are created under `work/models/`. For example:

```bash
python predict.py \
  --input queries.fasta \
  --model-dir work/models/full_length/LucaVirus_SVM \
  --length full_length \
  --lucavirus-tasks external/LucaVirusTasks \
  --output predictions.tsv
```

For a fixed-length model, the fitting pipeline stores its deterministic window seed when regenerated; otherwise pass `--slice-seed SEED` explicitly. Ensure queries are long enough. The query input does not become training data. Predicted class and `prob_Insect`, `prob_Plant`, `prob_Other` are written to TSV. Do not automatically choose a length model without a validated routing policy.

Saved classifiers are joblib/pickle-based artifacts; load only files generated locally or distributed from an independently verified trusted source.
