# Phase 03: Paper — Context

## Phase Goal
A submission-ready IEEE TCCN/TWC journal manuscript with all required sections, publication-quality figures, and reproducibility support.

## Requirements

- **PAPER-01**: Complete IEEEtran journal LaTeX document structure (double-column, all sections)
- **PAPER-02**: Introduction section — motivation, contributions, paper organization
- **PAPER-03**: Related work section — adversarial attacks on AMC, input-transformation defenses, classical filtering
- **PAPER-04**: System model and threat model section — AWN architecture, attack models, defense pipeline
- **PAPER-05**: Proposed method section — spectral-gated defense algorithm, adaptive-K, classical filter baselines
- **PAPER-06**: Experimental setup section — dataset, model, attack configs, defense parameters, evaluation metrics
- **PAPER-07**: Results and analysis section — defense comparison tables, confusion matrices, budget curves, latency
- **PAPER-08**: Publication-quality figures — accuracy-vs-SNR curves, confusion matrix heatmaps, defense comparison bar charts, frequency-domain spectra, budget curve plots
- **PAPER-09**: Conclusion section
- **PAPER-10**: Reproducibility script — single shell script to regenerate all CSVs and figures

## Success Criteria (from ROADMAP)

1. A complete IEEEtran journal LaTeX manuscript compiles without errors and covers all required sections (introduction, related work, system model and threat model, proposed method, experimental setup, results and analysis, conclusion)
2. All figures (accuracy-vs-SNR curves, confusion matrices, defense comparison bar charts, frequency-domain spectra) render as PDF at 300 dpi with IEEE-matching fonts and are referenced from the manuscript
3. A single shell script re-runs all experiments from raw data and regenerates all result CSVs and figures referenced in the paper

## Depends On

- Phase 1: Defense implementations (all defenses exist in util/defense.py, util/baseline_filters.py)
- Phase 2: Experimental results (all CSVs and .npy files in inference/2016.10a_165/result/defense_compare/)

## Existing Assets

### Paper Drafts
- `paper/latex/spectral_gated_defense_usenix.tex` — Complete USENIX-format paper (~680 lines) covering spectral-gated defense with sections: intro, background/threat model, control-plane analysis, defense algorithm, evaluation, FEC, receiver pipeline, discussion, related work, conclusion. Has inline BibTeX with 10 references.
- `paper/latex/crc_experiment_ieee.tex` — IEEE conference paper about CRC/AMC interaction under CW attacks
- `paper/latex/crc_experiment_content.tex` — Content sections for CRC paper

### Reports
- `reports/adaptive_k_report.md` — Detailed threat model with 3 deployment contexts (ITU/FCC monitoring, CBRS SAS, military ESM), adversary model, adaptive-K defense rationale

### Thesis Template
- `NYCU-thesis-template/` — Full thesis with sections: Introduction, Related Work, Architecture, Methodology, Evaluation, Conclusion

### Experimental Data (Phase 2 outputs)
Located in `inference/2016.10a_165/result/defense_compare/`:
- `defense_compare.csv` — 495 rows: 9 defenses x 5 attacks x 10 SNRs + weighted averages
- `defense_compare_cw.csv`, `_eadl1.csv`, `_eaden.csv`, `_fgsm.csv`, `_pgd.csv` — Per-attack pivot tables
- `confmat/*.npy` — 18 confusion matrices (cw/eadl1/eaden x SNR 0/10/18 x before/after)
- `confmat/confmat_summary.csv` — Summary with accuracy before/after defense
- `budget_curves/budget_curves_detail.csv`, `budget_curves_agg.csv` — Perturbation budget data
- `budget_curves/budget_fgsm.csv`, `budget_pgd.csv`, `budget_cw.csv`, `budget_eadl1.csv`, `budget_eaden.csv`

### Calibration
- `inference/2016.10a_165/result/calibration_params.json` — Per-SNR best params for 5 classical filters

### Existing Figures
- `paper/iq_constellation_clean/` — Clean IQ constellation plots for all 11 modulations
- `paper/iq_constellation_normalized/` — Normalized constellation plots
- `paper/iq_fgsm_bpsk/` — FGSM attack effect on BPSK constellation

### Key Experimental Results (Phase 2 Summary)
Defense comparison weighted averages:
| Defense | CW | EAD-L1 | EAD-EN | FGSM | PGD |
|---|---|---|---|---|---|
| adaptive_k | 77.4% | 79.8% | 79.4% | 64.6% | 60.8% |
| spectral_gated | 76.1% | 77.0% | 76.6% | 63.4% | 59.4% |
| no_defense | 75.3% | 75.6% | 75.3% | 62.9% | 57.9% |
| kalman | 72.7% | 72.5% | 71.9% | 63.2% | 59.7% |
| wiener | 72.3% | 72.4% | 71.9% | 63.1% | 60.1% |
| gaussian | 72.3% | 72.2% | 71.8% | 62.4% | 58.8% |
| fir | 71.1% | 71.4% | 71.1% | 62.9% | 61.3% |
| savitzky_golay | 70.7% | 70.9% | 70.7% | 62.1% | 59.3% |
| rand_smooth | 40.8% | 41.2% | 41.2% | 40.4% | 39.9% |

## Constraints

- IEEE TCCN/TWC journal format (IEEEtran document class, journal option)
- LaTeX must compile with standard tools (pdflatex + bibtex)
- Figures as PDF/EPS at 300+ dpi with IEEE-matching fonts (Computer Modern or Times)
- Target length: 10-14 double-column pages
- Must reference all Phase 2 data artifacts
- Python figure-generation scripts should use matplotlib with IEEE-compatible styling

## Key Decisions Needed

- Whether to include CRC/FEC analysis from USENIX draft (expands scope but strengthens contribution)
- Whether to include control-plane attack analysis (same consideration)
- Bibliography: expand inline BibTeX from USENIX draft or maintain separate .bib file
- Figure layout: which results to present as tables vs. figures
