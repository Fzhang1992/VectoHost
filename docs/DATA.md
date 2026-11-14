# Input data

The pipeline accepts three multi-record FASTA files, one per host-category label. The three inputs must contain the **already curated, one-record-per-virus-TaxID** full-length nucleotide sequences.

| Repository file | FASTA path used by the pipeline |
|---|---|
| [`Insect_full_length.zip`](https://github.com/Fzhang1992/VectoHost/blob/main/Insect_full_length.zip) | `input/full_length/Insect_full_length.fasta` |
| [`Plant_full_length.fasta`](https://github.com/Fzhang1992/VectoHost/blob/main/Plant_full_length.fasta) | `input/full_length/Plant_full_length.fasta` |
| [`Other_full_length.zip`](https://github.com/Fzhang1992/VectoHost/blob/main/Other_full_length.zip) | `input/full_length/Other_full_length.fasta` |

`Insect_full_length.zip` contains `Insect_full_length.fasta`. `Other_full_length.zip` contains `full_length/Other_full_length.fasta` and uses Git LFS, so a Git clone must fetch the actual archive rather than retaining its small pointer file. From the repository root, run:

```bash
mkdir -p input/full_length
cp Plant_full_length.fasta input/full_length/
unzip -o Insect_full_length.zip -d input/full_length/
unzip -o Other_full_length.zip -d input/
```

The default `INSECT_FASTA`, `PLANT_FASTA`, and `OTHER_FASTA` values in `config.example.env` match these extracted paths. ZIP archives are not supported as direct inputs; plain `.fasta` and gzip-compressed FASTA are accepted. Headers must start with a unique numeric virus TaxID, such as `>123456|virus=name`; TaxIDs must not be duplicated across input classes. The scripts do not select strain accessions, align genomic segments, or redownload Virus-Host DB: those decisions must be resolved before the three input collections are created.

## Research collection sizes

| Dataset | Insect | Plant | Other |
|---|---:|---:|---:|
| full length and 1,000 bp | 1,040 | 2,111 | 30,314 |
| 3,000 bp | 927 | 1,511 | 29,494 |
| 5,000 bp | 838 | 1,340 | 28,313 |

For the final **paper metrics**, 14 pre-identified TaxIDs with documented insect and plant host associations were excluded from pooled OOF evaluation: 1,026 Insect records at 1,000 bp/full length, 914 at 3,000 bp, and 826 at 5,000 bp. They do not enter the training folds; historical runs may include them in separately reserved validation folds before the final exclusion.

The other two classes retain their full counts. Validation checks fail if class-specific FASTAs, TaxIDs or lengths do not match the requested research profile (`EXPECT_PUBLISHED=1`). For new biological datasets, set it to 0.

## Sequence windows

- `regenerate`: generate one deterministic, contiguous fragment of the requested size per eligible full-length sequence, using an explicit SHA256 seed. The full-length group is unchanged.
- `historical`: read frozen original 1,000/3,000/5,000-bp FASTA fragments; feature matrices are still **newly computed**. This mode needs the historical fragment files as well as three full-length FASTAs.

Use `historical` when original sequence-window identity is essential to reproduce published metrics. A new `regenerate` run is a new sampling of fragments unless its sequence hashes match the published material.
