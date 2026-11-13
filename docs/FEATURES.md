# Fresh feature extraction

Each length and class runs **all** of these extraction methods from the emitted sequence FASTA:

| Representation | Dimension | Method |
|---|---:|---|
| 1-mer | 4 | normalized counts of A, C, G, T |
| 2-mer | 16 | normalized overlapping adjacent dinucleotides |
| 3-mer | 64 | normalized overlapping trinucleotide frequencies |
| 4-mer | 256 | normalized overlapping tetranucleotide frequencies |
| ESM-2 | 320 | translate both strands in six reading frames; longest stop-free ORF, fallback when <50 aa; crop 1022 aa; `esm2_t6_8M_UR50D`, layer 6 mean pooling excluding BOS/EOS |
| LucaVirus | 2560 | official `LucaVirusTasks` inference on nucleotide FASTA; v1.0; 20240815023346; 3800000; `--seq_type gene --embedding_type vector --embedding_complete --embedding_fixed_len_a_time 4096` |

`src/vh3/embedding_lucavirus.py` invokes the upstream `src/llm/lucavirus/get_embedding.py` to generate **fresh** `vector_<TaxID>.pt` files, verifies complete TaxID mapping, and converts them to `features.npy` in FASTA order. It refuses missing files/nonfinite vectors/other dimensions. The model checkpoint must be available via upstream official setup. For security, only locally produced trusted `.pt` vector files are loaded.

`src/vh3/embedding_esm2.py` uses fair-esm and records an available checkpoint SHA256 and ORF decisions, so the original extraction can be audited. Numerical equivalence to the three-class historical ESM-2 vectors still requires an original-data run.

**There is no `--reuse-embedding` or precomputed-feature import stage.** Every real run re-generates each representation from sequence data. Model fitting standardization is implemented inside the appropriate scikit-learn pipelines, and the validation fold is transformed using training-fold fitted parameters.
