# Installation and pretrained models

This project targets Linux. An isolated Conda environment is recommended because upstream LucaVirusTasks has its own Python dependencies.

```bash
conda env create -f environment.yml
conda activate virushost3class
cp config.example.env config.env
bash scripts/setup_models.sh
```

`setup_models.sh` checks out the official [LucaVirusTasks](https://github.com/LucaOne/LucaVirusTasks) source under `external/LucaVirusTasks` and installs its dependencies. If you have an already verified checkout, set `LUCAVIRUS_TASKS` to its path and set `LUCAVIRUS_COMMIT` to the appropriate revision.

## Foundation model weights

- **ESM-2:** `fair-esm` loads the official `esm2_t6_8M_UR50D` weights, typically downloading them on first use. The script records the cached checkpoint SHA256 where available.
- **LucaVirus:** Use the official LucaVirus / LucaVirusTasks checkpoint mechanism for the **v1.0 / 20240815023346 / step 3800000** foundation model. See the upstream [foundation model README](https://github.com/LucaOne/LucaVirus) and [tasks README](https://github.com/LucaOne/LucaVirusTasks). Some upstream scripts support automatic checkpoint downloading; they require network and disk space. The optional official checkpoint-fetcher may download *additional models*; do not run it unless necessary.

To use upstream's bulk checkpoint download explicitly:

```bash
bash scripts/setup_models.sh --fetch-checkpoints
```

Large foundation weights are **not** provided by this repository. The `LUCAVIRUS_CHECKPOINT_PATH` and `LUCAVIRUS_CHECKPOINT_SHA256` fields in `config.env` allow additional verification of a known file; they do not change where upstream code loads checkpoints. For exact paper-level numerical comparison, also pin the upstream git commit and preserve the checkpoints' hashes.

## Test without large models

```bash
python scripts/example_run.py
PYTHONPATH=src python -m pytest -q tests
```

This works on synthetic data. `bash run_all.sh --check` verifies a configured full-data environment but does not perform a real ESM-2 or LucaVirus forward pass.
