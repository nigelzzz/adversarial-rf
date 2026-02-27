# CRC Experiment Paper Templates (IEEE / ACM)

This folder provides two LaTeX wrappers that share the same paper content:

- `crc_experiment_ieee.tex` (IEEEtran)
- `crc_experiment_acm.tex` (ACM acmart)

Shared files:

- `crc_experiment_content.tex` (sections, tables, discussion)
- `crc_experiment_refs.tex` (references)

## Build

Run from this directory:

```bash
cd paper/latex
pdflatex crc_experiment_ieee.tex
pdflatex crc_experiment_ieee.tex
```

For ACM:

```bash
cd paper/latex
pdflatex crc_experiment_acm.tex
pdflatex crc_experiment_acm.tex
```

## Notes

- Update title, author metadata, and venue fields before submission.
- Numeric results are sourced from `crc_experiment_results/crc_vs_amc.csv`.
