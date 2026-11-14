# Fitted models

Run `bash run_all.sh --run` with the three complete training FASTAs to fit four independent LucaVirus+SVM classifiers. Their outputs are written to `work/models/<length>/LucaVirus_SVM/`.

The resulting `model.joblib` files are **not included in the source repository**. Each includes a SHA256-protected model manifest. Only load joblib files from trusted sources. Underlying ESM-2 and LucaVirus foundation-model weights remain subject to their respective licenses.
