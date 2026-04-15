---
phase: 03-paper
verified: 2026-04-06T13:30:00Z
status: gaps_found
score: 9/10 must-haves verified
gaps:
  - truth: "All figures (accuracy-vs-SNR curves, confusion matrices, defense comparison bar charts, frequency-domain spectra) render as PDF at 300 dpi with IEEE-matching fonts and are referenced from the manuscript"
    status: partial
    reason: "freq_spectra_cw.pdf is generated using placeholder synthetic spectra (not real CW attack data) because model/dataset loading failed at figure generation time. More critically, freq_spectra_cw.pdf is not referenced via \\includegraphics from any section in the manuscript — it exists in paper/latex/figures/ but is orphaned from the paper body."
    artifacts:
      - path: "paper/latex/figures/freq_spectra_cw.pdf"
        issue: "Uses synthetic placeholder spectra (not real CW attack data). Not referenced from any \\includegraphics in the paper sections. PAPER-09 (frequency-domain visualization) is only partially satisfied — the figure exists but is not in the manuscript."
    missing:
      - "Either add \\includegraphics{freq_spectra_cw} to results.tex (or proposed_method.tex) with a figure environment and caption, OR regenerate the figure with real data from the running environment and add it to the paper."
human_verification:
  - test: "Compile paper/latex/main.tex with latexmk -pdf from a clean build and check the compiled PDF visually"
    expected: "13-page IEEEtran double-column PDF with all figures rendering correctly, no missing figure boxes, all fonts matching IEEEtran style"
    why_human: "latexmk reports 'nothing to do' and PDF already exists; visual inspection of figure layout (subfigure alignment, column widths, confusion matrix readability at 300 dpi) requires human review"
  - test: "Run bash paper/reproduce.sh --figures from repo root with the Python venv activated"
    expected: "Script completes with 11 PDFs regenerated in paper/latex/figures/ and latexmk compiles the paper to a valid PDF"
    why_human: "Cannot activate the venv and run the full figure pipeline safely without risking overwriting valid figures with placeholder versions; requires human to verify the --figures path works end-to-end"
---

# Phase 03: Paper Verification Report

**Phase Goal:** A submission-ready IEEE TCCN/TWC manuscript with all required sections, figures, and reproducibility support
**Verified:** 2026-04-06T13:30:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A complete IEEEtran journal LaTeX manuscript compiles without errors and covers all 7 required sections | VERIFIED | `latexmk` reports all targets up-to-date; main.pdf exists (13 pages, 506 KB); all 7 \\input{sections/...} present in main.tex; all 7 section files contain \section{} headers |
| 2 | All figures render as PDF at 300 dpi with IEEE-matching fonts and are referenced from the manuscript | PARTIAL | 10 of 11 figures are referenced from results.tex via \\includegraphics; freq_spectra_cw.pdf exists but is ORPHANED (not referenced from any section file); additionally, freq_spectra_cw.pdf uses synthetic placeholder spectra, not real CW attack data |
| 3 | A single shell script re-runs all experiments from raw data and regenerates all result CSVs and figures referenced in the paper | VERIFIED | paper/reproduce.sh exists (169 lines, executable, syntax-valid); contains preflight checks, 5-step pipeline (calibrate_defenses → defense_compare → adv_eval → generate_figures → latexmk), --figures flag to skip expensive steps |

**Score:** 9/10 individual artifact checks pass (freq_spectra_cw.pdf fails: orphaned + placeholder content)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `paper/latex/main.tex` | IEEEtran journal master with \\documentclass[journal]{IEEEtran} | VERIFIED | Present, correct documentclass, complete abstract (200 words with specific numbers: 77.4%, 79.8%, 4.7pp, 7.3pp margins), 7 \\input sections, \\bibliography{refs} |
| `paper/latex/refs.bib` | 35+ BibTeX references, ≥300 lines | VERIFIED | 41 BibTeX entries (grep '@' = 41), 422 lines (exceeds 300-line minimum) |
| `paper/scripts/ieee_style.py` | Exports IEEE_STYLE, IEEE_COL_WIDTH, IEEE_DBL_WIDTH, apply_ieee_style | VERIFIED | All exports confirmed: IEEE_COL_WIDTH=3.487, IEEE_DBL_WIDTH=7.16, IEEE_STYLE dict (18 keys), apply_ieee_style/single_col_fig/double_col_fig callable, DEFENSE_COLORS (9 defenses), DEFENSE_LABELS |
| `paper/scripts/generate_figures.py` | Reads Phase 2 CSVs/NPYs, produces all paper figures, ≥200 lines | VERIFIED | 536 lines; reads defense_compare_*.csv via pd.read_csv, confmat *.npy via np.load, imports from ieee_style; each figure is a separate function |
| `paper/latex/figures/defense_compare_overview.pdf` | Grouped bar chart, 9 defenses × 5 attacks | VERIFIED | Exists (20,550 bytes); referenced from results.tex line 69 |
| `paper/latex/figures/acc_vs_snr_cw.pdf` | Accuracy vs SNR for CW attack | VERIFIED | Exists (21,851 bytes); referenced from results.tex line 86 |
| `paper/latex/figures/acc_vs_snr_eadl1.pdf` | Accuracy vs SNR for EAD-L1 attack | VERIFIED | Exists (21,816 bytes); referenced from results.tex line 91 |
| `paper/latex/figures/confmat_cw_snr18.pdf` | Before/after confusion matrix for CW at SNR=18 | VERIFIED | Exists (29,091 bytes); referenced from results.tex line 173 |
| `paper/latex/figures/confmat_eadl1_snr18.pdf` | Before/after confusion matrix for EAD-L1 at SNR=18 | VERIFIED | Exists (29,473 bytes); referenced from results.tex line 184 |
| `paper/latex/figures/budget_fgsm.pdf` | Perturbation budget curve for FGSM | VERIFIED | Exists (24,349 bytes); referenced from results.tex line 217 |
| `paper/latex/figures/budget_cw.pdf` | Perturbation budget curve for CW | VERIFIED | Exists (20,948 bytes); referenced from results.tex line 222 |
| `paper/latex/figures/freq_spectra_cw.pdf` | Frequency-domain spectra: clean/CW-attacked/Adaptive-K-recovered | ORPHANED + STUB | Exists (20,967 bytes) but: (1) uses synthetic placeholder spectra because model/dataset import failed at generation time; (2) not referenced from any \\includegraphics in any section file |
| `paper/latex/sections/introduction.tex` | ≥80 lines, 4 contributions, 8+ citations | VERIFIED | 97 lines, 4 numbered contributions in enumerate environment, 11 \\cite commands |
| `paper/latex/sections/related_work.tex` | ≥120 lines, 3 subsections, 20+ citations | VERIFIED | 124 lines, 3 subsections (A: adversarial AMC, B: defense methods, C: classical filters), 22 \\cite commands |
| `paper/latex/sections/system_model.tex` | ≥120 lines, 4 subsections with attack equations | VERIFIED | 148 lines, 4 subsections (signal representation, AWN architecture, attack models, threat model), attack objectives with math equations |
| `paper/latex/sections/proposed_method.tex` | ≥150 lines, 2 algorithm environments | VERIFIED | 279 lines, 6 subsections (A-F), 2 algorithm environments (Algorithm 1: Spectral-Gated, Algorithm 2: Adaptive-K) |
| `paper/latex/sections/experimental_setup.tex` | ≥80 lines, attack config table, defense config table | VERIFIED | 132 lines, 4 subsections (A-D), 2 \\begin{table} environments (attack params, defense params) |
| `paper/latex/sections/results.tex` | ≥200 lines, main comparison table, ≥4 figure inclusions | VERIFIED | 307 lines, 7 subsections, 1 \\begin{table*} (9-defense × 5-attack comparison with bold best-per-column), 9 \\includegraphics references covering all attack types and figure types |
| `paper/latex/sections/conclusion.tex` | ≥30 lines, 4 findings, future work | VERIFIED | 62 lines, 4-item enumerate with labeled findings including specific percentage margins, 3 future work sentences |
| `paper/reproduce.sh` | ≥30 lines, executable, references main.py defense_compare and generate_figures.py | VERIFIED | 169 lines, executable (-rwxrwxr-x), syntax valid (bash -n passes), contains `python main.py --mode calibrate_defenses`, `python main.py --mode defense_compare`, `python paper/scripts/generate_figures.py`, latexmk compilation |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `paper/scripts/generate_figures.py` | `inference/2016.10a_165/result/defense_compare/` | `pd.read_csv` and `np.load` | WIRED | Lines 93, 141: `pd.read_csv(path, index_col='defense')`; lines 191-192: `np.load(before_path/after_path).astype(float)` |
| `paper/scripts/generate_figures.py` | `paper/scripts/ieee_style.py` | `from ieee_style import` | WIRED | Line 28: `from ieee_style import (IEEE_COL_WIDTH, IEEE_DBL_WIDTH, ...)` — sys.path patched at line 26 |
| `paper/latex/main.tex` | `paper/latex/figures/` | `\includegraphics` in section files | PARTIAL | main.tex has `\graphicspath{{figures/}}`; results.tex references 9 figures via \\includegraphics; freq_spectra_cw.pdf is NOT referenced from any section |
| `paper/latex/main.tex` | `paper/latex/refs.bib` | `\bibliography{refs}` | WIRED | Line 87: `\bibliography{refs}`; main.log shows 0 undefined references |
| `paper/latex/sections/introduction.tex` | `paper/latex/refs.bib` | `\cite{}` commands | WIRED | 11 \\cite commands verified; 0 undefined citation warnings in main.log |
| `paper/latex/sections/related_work.tex` | `paper/latex/refs.bib` | `\cite{}` commands referencing 20+ papers | WIRED | 22 \\cite commands verified |
| `paper/latex/sections/proposed_method.tex` | `paper/latex/figures/` | algorithm environments | WIRED | 2 algorithm environments (alg:gated, alg:adaptivek) referenced in text via \\ref |
| `paper/reproduce.sh` | `main.py` | `python main.py --mode defense_compare` | WIRED | Lines 81-104: `python main.py --mode calibrate_defenses` and `python main.py --mode defense_compare --dataset ... --ckpt_path ... --max_per_cell 200` |
| `paper/reproduce.sh` | `paper/scripts/generate_figures.py` | `python paper/scripts/generate_figures.py` | WIRED | Line 131: `python paper/scripts/generate_figures.py` |
| `paper/latex/sections/results.tex` | `inference/2016.10a_165/result/defense_compare/` | Numbers hardcoded from CSV data | WIRED | Line 23: 77.40, 79.80 match defense_compare.csv adaptive_k weighted averages; line 43-44: "77.4% on CW vs. 72.7% for Kalman" matches CSV |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `paper/latex/sections/results.tex` | Defense comparison table numbers | `inference/2016.10a_165/result/defense_compare/defense_compare_*.csv` | Yes — real Phase 2 experimental results; values like 77.40, 79.80, 72.7 match the CSV contents | FLOWING |
| `paper/scripts/generate_figures.py` | acc_vs_snr plots | `defense_compare_{attack}.csv` via pd.read_csv | Yes — reads per-attack pivot tables with real SNR×defense accuracy data | FLOWING |
| `paper/scripts/generate_figures.py` | confmat heatmaps | `confmat/{attack}_snr{snr}_{before|after}.npy` via np.load | Yes — loads actual 11×11 confusion matrices from Phase 2 experiments | FLOWING |
| `paper/latex/figures/freq_spectra_cw.pdf` | 3-panel FFT spectra | Falls back to `_plot_freq_spectra_placeholder()` (synthetic QPSK-like signals) | No — uses synthesized data when dataset/model unavailable at generation time | STATIC |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| ieee_style.py exports required constants and functions | `python -c "import ieee_style; print(len(ieee_style.IEEE_STYLE))"` | 18 keys; IEEE_COL_WIDTH=3.487, DEFENSE_COLORS has 9 keys, all callables verified | PASS |
| generate_figures.py module loads without import errors | `python -c "import importlib.util; spec=importlib.util.spec_from_file_location(...); print('OK')"` | Module loads without error | PASS |
| LaTeX compiles without errors | `cd paper/latex && latexmk -pdf -interaction=nonstopmode main.tex` | "All targets (main.pdf) are up-to-date" — 0 undefined references in main.log | PASS |
| PDF is 13 pages (within 12-14 page target) | `pdfinfo paper/latex/main.pdf \| grep Pages` | Pages: 13 | PASS |
| reproduce.sh has valid bash syntax | `bash -n paper/reproduce.sh` | Exit code 0, syntax OK | PASS |
| reproduce.sh is executable | `test -x paper/reproduce.sh` | File is executable | PASS |
| 11 figure PDFs exist with non-zero size | `ls paper/latex/figures/*.pdf \| wc -l` | 11 files, smallest 20,550 bytes | PASS |
| results.tex references 9+ figures | `grep -c 'includegraphics' sections/results.tex` | 9 occurrences | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PAPER-01 | 03-02 | IEEE TCCN LaTeX manuscript — Introduction section | SATISFIED | introduction.tex: 97 lines, 4 contributions, 11 citations |
| PAPER-02 | 03-02 | IEEE TCCN LaTeX manuscript — Related Work section | SATISFIED | related_work.tex: 124 lines, 3 subsections, 22 citations |
| PAPER-03 | 03-02 | IEEE TCCN LaTeX manuscript — System Model & Threat Model section | SATISFIED | system_model.tex: 148 lines, 4 subsections including AWN equations and attack objectives |
| PAPER-04 | 03-02 | IEEE TCCN LaTeX manuscript — Proposed Defense Method section | SATISFIED | proposed_method.tex: 279 lines, 6 subsections, 2 algorithm environments |
| PAPER-05 | 03-02 | IEEE TCCN LaTeX manuscript — Experimental Setup section | SATISFIED | experimental_setup.tex: 132 lines, 4 subsections, 2 parameter tables |
| PAPER-06 | 03-03 | IEEE TCCN LaTeX manuscript — Results & Analysis section | SATISFIED | results.tex: 307 lines, 7 subsections, 9×5 comparison table, 9 figure references, exact numbers from Phase 2 CSV |
| PAPER-07 | 03-03 | IEEE TCCN LaTeX manuscript — Conclusion section | SATISFIED | conclusion.tex: 62 lines, 4 numbered findings with percentages, 3 future work directions |
| PAPER-08 | 03-01 | Publication-quality figures (accuracy curves, confusion matrices, spectral plots) | SATISFIED | 11 PDF figures at IEEE dimensions (3.487in/7.16in) at 300 dpi; 10 of 11 are referenced from the manuscript |
| PAPER-09 | 03-01, 03-03 | Frequency-domain visualization plots (clean→attacked→recovered spectra) | BLOCKED | freq_spectra_cw.pdf exists (20,967 bytes) but: (a) uses synthetic placeholder spectra rather than real CW attack data, (b) is NOT referenced via \\includegraphics from any section in the manuscript body |
| PAPER-10 | 03-03 | Reproducibility scripts to regenerate all experimental results | SATISFIED | paper/reproduce.sh: 169 lines, executable, 5-step pipeline, --figures flag, preflight checks, references main.py and generate_figures.py correctly |

**Orphaned requirements check:** No additional PAPER-* requirements mapped to Phase 3 in REQUIREMENTS.md beyond the declared 10.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `paper/scripts/generate_figures.py` | 419, 422 | `_plot_freq_spectra_placeholder()` fallback for freq_spectra_cw | Warning | freq_spectra_cw.pdf contains synthetic QPSK-like spectra, not real CW attack data; the figure is illustrative-only and does not represent experimental results |
| `paper/scripts/generate_figures.py` | 311-419 | try/except wraps entire `plot_freq_spectra_cw()` | Info | Design choice: graceful degradation when model/dataset unavailable; documented in 03-01-SUMMARY Known Stubs |
| `paper/latex/main.tex` | 85 | `\nocite{*}` — forces all 41 refs into bibliography even if uncited | Warning | All 41 bibliography entries appear in the compiled PDF regardless of whether they are cited. In a submission, uncited references should be removed. This was noted in 03-01-SUMMARY as intentional for the stub phase and in 03-03-SUMMARY as "kept in". Reviewers may notice references listed but not cited. |

---

### Human Verification Required

#### 1. Visual Figure Quality Inspection

**Test:** Open `paper/latex/main.pdf` and visually inspect each figure
**Expected:** Confusion matrix heatmaps (confmat_cw_snr18.pdf etc.) show readable 11×11 grids with modulation labels at ≥5pt font; line plots (acc_vs_snr_*.pdf) show distinguishable defense lines with a readable legend; grouped bar chart (defense_compare_overview.pdf) shows all 9 defense groups with correct attack groupings
**Why human:** Automated tools cannot assess readability at IEEE print dimensions, whether labels overlap, or whether the greyscale-distinguishable colors print correctly

#### 2. Full reproduce.sh End-to-End Run (Figures Path)

**Test:** In the project venv, run `bash paper/reproduce.sh --figures` from the repo root
**Expected:** Script completes without error, 11 PDFs are regenerated in paper/latex/figures/, latexmk compiles main.pdf successfully
**Why human:** Cannot safely activate the venv and run the full pipeline without overwriting existing figures; also verifies that the `plot_freq_spectra_cw()` function either successfully loads the model/data (replacing the placeholder) or gracefully falls back

#### 3. \nocite{*} Submission Check

**Test:** Verify that all 41 BibTeX entries in refs.bib are actually cited with \\cite{} commands across the 7 section files, or identify which entries are uncited
**Expected:** All entries cited; or uncited entries removed before submission
**Why human:** Counting unique \\cite{} keys vs. \\@-entries in refs.bib requires cross-referencing multiple files; impact is submission quality, not compilation correctness

---

## Gaps Summary

One gap blocks full success-criteria satisfaction. The `freq_spectra_cw.pdf` figure (PAPER-09) has two issues:

1. **Not referenced from the manuscript.** The figure was generated by `generate_figures.py` and exists in `paper/latex/figures/` but no section file contains an `\includegraphics{freq_spectra_cw}` command. The PLAN-01 must_haves listed this as an artifact to produce, and the phase goal's second success criterion requires figures to be "referenced from the manuscript."

2. **Placeholder content.** When `generate_figures.py` runs, `plot_freq_spectra_cw()` attempts to load the AWN model and dataset via the project's venv. When that fails (e.g., in a context without the venv or dataset), it silently falls back to `_plot_freq_spectra_placeholder()` with synthetic QPSK-like spectra. This is documented as a known limitation in the 03-01-SUMMARY, but it means the frequency-domain figure does not show real CW adversarial data.

The other 9 success-criterion elements all pass: the manuscript compiles to 13 pages with 0 undefined references, all 10 other figures are substantive and referenced, the bibliography has 41 complete entries, all 7 sections are complete with appropriate content and line counts, and reproduce.sh is a complete end-to-end pipeline.

Two additional non-blocking observations:
- `\nocite{*}` in main.tex forces all 41 references into the compiled PDF; this should be removed and replaced with explicit \\cite{} commands before submission, or any uncited entries should be removed.
- The success criterion requiring "300 dpi with IEEE-matching fonts" notes that `text.usetex=False` in ieee_style.py means figures use matplotlib's DejaVu/serif font fallback rather than actual Computer Modern Roman. The decision was documented in 03-01-SUMMARY. For a camera-ready submission, enabling `text.usetex=True` (with a working LaTeX installation in the Python environment) would produce exact IEEE font matching.

---

_Verified: 2026-04-06T13:30:00Z_
_Verifier: Claude Sonnet 4.6 (gsd-verifier)_
