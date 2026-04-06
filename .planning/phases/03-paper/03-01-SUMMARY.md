---
phase: 03-paper
plan: "01"
subsystem: paper-infrastructure
tags: [latex, figures, bibliography, matplotlib, ieee]
dependency_graph:
  requires: [02-experimental-results]
  provides: [paper-latex-infrastructure, figure-pdfs, bibliography]
  affects: [03-02, 03-03]
tech_stack:
  added: []
  patterns:
    - IEEEtran journal double-column LaTeX document structure
    - Shared matplotlib IEEE style module (ieee_style.py) with DEFENSE_COLORS palette
    - generate_figures.py single-script figure regeneration from Phase 2 CSVs/NPYs
key_files:
  created:
    - paper/latex/main.tex
    - paper/latex/refs.bib
    - paper/latex/sections/introduction.tex
    - paper/latex/sections/related_work.tex
    - paper/latex/sections/system_model.tex
    - paper/latex/sections/proposed_method.tex
    - paper/latex/sections/experimental_setup.tex
    - paper/latex/sections/results.tex
    - paper/latex/sections/conclusion.tex
    - paper/scripts/ieee_style.py
    - paper/scripts/generate_figures.py
    - paper/latex/figures/defense_compare_overview.pdf
    - paper/latex/figures/acc_vs_snr_cw.pdf
    - paper/latex/figures/acc_vs_snr_eadl1.pdf
    - paper/latex/figures/acc_vs_snr_fgsm.pdf
    - paper/latex/figures/acc_vs_snr_pgd.pdf
    - paper/latex/figures/confmat_cw_snr18.pdf
    - paper/latex/figures/confmat_eadl1_snr18.pdf
    - paper/latex/figures/confmat_eaden_snr18.pdf
    - paper/latex/figures/budget_fgsm.pdf
    - paper/latex/figures/budget_cw.pdf
    - paper/latex/figures/freq_spectra_cw.pdf
  modified: []
decisions:
  - "text.usetex=False in IEEE_STYLE to avoid pdflatex-in-Python dependency; serif font falls back to DejaVu with findfont warnings (cosmetic, PDFs are valid)"
  - "nocite{*} in main.tex during stub phase so bibliography compiles with all 41 refs; will be replaced by explicit cites in Plan 02"
  - "freq_spectra_cw uses placeholder synthetic spectra when model/dataset not found; will be regenerated with real CW attack in Plan 02"
  - "Budget plot uses _COMPACT_DEFENSES subset to reduce legend clutter on single-col figure"
metrics:
  duration_minutes: 6
  completed_date: "2026-04-06"
  tasks_completed: 2
  tasks_total: 2
  files_created: 22
  files_modified: 0
---

# Phase 03 Plan 01: LaTeX Infrastructure and Figure Pipeline Summary

IEEEtran journal document with 7 section stubs, 41-entry bibliography, IEEE matplotlib styling module, and single-script figure pipeline generating 11 paper PDFs from Phase 2 data.

## What Was Built

### Task 1: LaTeX Document Structure

- **`paper/latex/main.tex`**: IEEEtran journal double-column master document with `\documentclass[journal]{IEEEtran}`, full preamble (amsmath, booktabs, graphicx, algorithm, subcaption, hyperref), placeholder abstract, 7 section inputs, and bibliography. Compiles to 2-page PDF with latexmk.
- **7 section stub files** in `paper/latex/sections/`: Each contains the section header, subsection stubs with `% TODO: content in Plan 02` comments. All section stubs are present and compilable.
- **`paper/scripts/ieee_style.py`**: Shared matplotlib configuration module exporting `IEEE_STYLE` dict (18 rcParams), `IEEE_COL_WIDTH=3.487`, `IEEE_DBL_WIDTH=7.16`, `apply_ieee_style()`, `single_col_fig()`, `double_col_fig()`, `DEFENSE_COLORS` (9 defenses), `DEFENSE_LABELS`, `DEFENSE_ORDER`, `ATTACK_LABELS`.
- **`paper/latex/refs.bib`**: 41 BibTeX entries spanning: AMC classifiers (6), adversarial attacks on AMC (6), general adversarial ML (6), defense methods (6), classical filtering (5), communications/RF fundamentals (5), deep learning fundamentals (3), additional (4).

### Task 2: Figure Generation Pipeline

- **`paper/scripts/generate_figures.py`**: 536-line script that reads Phase 2 CSVs/NPYs from `inference/2016.10a_165/result/defense_compare/` and produces 11 paper figures as PDFs. Each figure type is a separate function (`plot_defense_overview`, `plot_acc_vs_snr`, `plot_confmat`, `plot_budget`, `plot_freq_spectra_cw`). Handles missing data files gracefully.

**11 figure PDFs produced:**

| Figure | Type | Dimensions |
|--------|------|-----------|
| `defense_compare_overview.pdf` | Grouped bar (9 defenses x 5 attacks) | Double-column |
| `acc_vs_snr_cw.pdf` | Line plot (9 defense lines) | Single-column |
| `acc_vs_snr_eadl1.pdf` | Line plot | Single-column |
| `acc_vs_snr_fgsm.pdf` | Line plot | Single-column |
| `acc_vs_snr_pgd.pdf` | Line plot | Single-column |
| `confmat_cw_snr18.pdf` | Side-by-side heatmaps (11x11) | Double-column |
| `confmat_eadl1_snr18.pdf` | Side-by-side heatmaps | Double-column |
| `confmat_eaden_snr18.pdf` | Side-by-side heatmaps | Double-column |
| `budget_fgsm.pdf` | Budget curve (linear x-axis) | Single-column |
| `budget_cw.pdf` | Budget curve (log x-axis) | Single-column |
| `freq_spectra_cw.pdf` | 3-panel spectra (placeholder) | Double-column |

## Verification Results

1. `latexmk -pdf main.tex` — compiles without errors (2-page PDF)
2. `ls paper/latex/figures/*.pdf | wc -l` — 11 (>= 10 required)
3. `grep -c '@' paper/latex/refs.bib` — 41 (>= 35 required)
4. `ieee_style.py` exports `IEEE_STYLE` dict with 18 keys

## Deviations from Plan

**1. [Rule 2 - Missing critical functionality] `\nocite{*}` added to compile stub document**
- **Found during:** Task 1 verification (latexmk compilation)
- **Issue:** Empty bibliography causes `! LaTeX Error: Something's wrong--perhaps a missing \item` when no `\cite{}` commands exist in section stubs
- **Fix:** Added `\nocite{*}` before `\bibliographystyle` so all 41 refs are included during stub phase; will be removed and replaced by explicit cites in Plan 02
- **Files modified:** `paper/latex/main.tex`
- **Commit:** 3971e80

**2. [Rule 2 - Missing critical functionality] Placeholder freq spectra when model unavailable**
- **Found during:** Task 2 — `plot_freq_spectra_cw()` requires dataset + checkpoint + torchattacks
- **Issue:** System-level python3 lacks the project's torch/torchattacks environment
- **Fix:** Added `_plot_freq_spectra_placeholder()` function using synthetic QPSK-like spectrum when imports fail; figure still renders at correct IEEE dimensions with 3-panel layout
- **Files modified:** `paper/scripts/generate_figures.py`
- **Commit:** bd63e32

## Known Stubs

- `freq_spectra_cw.pdf`: Uses synthetic placeholder spectra (not real CW attack data). The figure shows the correct layout (3 panels: clean/attacked/recovered) but with simulated signal characteristics. Will be regenerated with real data when Plan 02 adds the reproducibility script with proper environment.
- `paper/latex/sections/*.tex`: All 7 section files contain `% TODO: content in Plan 02` stubs. This is intentional — content is Plan 02/03 scope.

## Self-Check: PASSED

Files created verified:
- `paper/latex/main.tex` exists
- `paper/scripts/ieee_style.py` exists
- `paper/scripts/generate_figures.py` exists
- `paper/latex/refs.bib` exists (41 entries)
- `paper/latex/figures/defense_compare_overview.pdf` exists
- `paper/latex/figures/acc_vs_snr_cw.pdf` exists
- `paper/latex/figures/confmat_cw_snr18.pdf` exists
- `paper/latex/figures/budget_fgsm.pdf` exists
- `paper/latex/figures/freq_spectra_cw.pdf` exists
- 11 total PDFs in paper/latex/figures/

Commits verified:
- `3971e80` — Task 1: LaTeX doc structure, ieee_style.py, refs.bib
- `bd63e32` — Task 2: generate_figures.py and 11 figure PDFs
