# Model methods

The workflow generates four length-specific datasets (1,000, 3,000, 5,000 bp and full length), each with one nucleotide sequence per eligible virus TaxID. Training labels are Insect = 0, Plant = 1, Other = 2.

Six feature representations are evaluated separately: normalized overlapping nucleotide 1-/2-/3-/4-mer frequencies (4, 16, 64, 256 dimensions), ESM-2 protein embeddings (320 dimensions), and LucaVirus nucleotide embeddings (2,560 dimensions).

For ESM-2, each input nucleotide sequence is translated in six reading frames; the longest stop-free amino-acid segment is selected, with the historical fallback when shorter than 50 residues, and capped at 1,022 residues. The 320-dimensional representation is the mean of layer-6 residue embeddings from `esm2_t6_8M_UR50D` (excluding BOS/EOS).

For LucaVirus, the upstream `LucaVirusTasks/src/llm/lucavirus/get_embedding.py` is called on the nucleotide FASTA with v1.0 checkpoint identifier `20240815023346/step3800000`, `--seq_type gene`, `--embedding_type vector`, `--embedding_complete`, and `--embedding_fixed_len_a_time 4096`. Generated `vector_<TaxID>.pt` files are indexed by TaxID, checked for finite 2,560-dimensional values, and written to the training matrix in original FASTA order. Existing embeddings from a previous study are **not** inputs.

Models: RF (`n_estimators=500`), KNN (`n_neighbors=5` with fold-fitted `StandardScaler`), RBF SVM (`C=1.0`, `gamma=scale`, `probability=True` and fold-fitted `StandardScaler`), and GaussianNB (`var_smoothing=1e-9`), implemented in scikit-learn. Five-fold `StratifiedKFold(shuffle=True, random_state=20260914)` is used. `RandomUnderSampler(sampling_strategy='all', random_state=20260914+fold)` is applied only within each training fold.

The 14 recorded dual-host TaxIDs are never used for training. All final metrics exclude them, including if they appeared in a reserved validation fold in historical runs. AUC is the unweighted mean of the three class-specific one-vs-rest AUC values from **pooled five-fold OOF predictions**, after rounding class probabilities to six decimals. Accuracy, precision, recall and F1 are also computed on the pooled non-dual-host OOF set. The final deployment models are independently fitted after cross-validation and must not be used to estimate their own test performance.

See [reproducibility](REPRODUCIBILITY.md) for historical-window and checkpoint limitations.
