# Milestones

## v1.0 Paper Submission Package (Shipped: 2026-04-15)

**Phases completed:** 3 phases, 11 plans, 18 tasks

**Key accomplishments:**

- Five classical signal-processing filter baselines (Kalman, Wiener, Savitzky-Golay, Gaussian, FIR) implemented in util/defense_baselines.py with GPU-native depthwise conv1d for Gaussian/FIR and scipy/numpy CPU paths for the others
- DEFENSE_REGISTRY dict with 8 entries, defend() detect->recover->classify pipeline with GPU/CPU latency breakdown, and randomized smoothing majority-vote classifier wrapper (k=20, sigma=0.01)
- Parameter calibration sweeps (PARAM_GRIDS with 84 combos), per-SNR composite-score grid search, torch.cuda.Event GPU / time.perf_counter CPU latency benchmark with 10-warmup, and PIPE-03 clean accuracy validation with 2% threshold in util/defense_calibrate.py
- Defense comparison evaluation framework: 9 defenses x 5 attacks x 10 SNR points with minmax torchattacks and per-modulation 200-sample cap, outputting per-attack pivot CSVs
- generate_confusion_matrices() added to util/defense_compare.py producing 18 raw .npy and row-normalized CSV confusion matrices (3 optimization attacks x 3 SNRs x before/after FFT Top-K defense) for Phase 3 figure rendering
- One-liner:
- Per-SNR calibrated filter params loaded from calibration_params.json and injected into cfg before each classical filter defense evaluation, with --mode calibrate_defenses entry point producing the JSON
- All GPU experiments completed: 9 defenses x 5 attacks x 10 SNR points, 18 confusion matrices, and perturbation budget curves
- `paper/latex/sections/introduction.tex`
- Three completed IEEE TCCN sections (experimental setup with 2 parameter tables, results with 9x5 comparison table and 9 figure inclusions, conclusion with 4 findings), updated abstract with specific accuracy numbers (77.4%/79.8% margins), and 169-line reproduce.sh with preflight checks and --figures flag; paper compiles to 13-page PDF with 0 undefined references.

---
