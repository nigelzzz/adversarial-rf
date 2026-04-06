#!/bin/bash
# reproduce.sh — Reproduce all experiments and figures for IEEE TCCN paper
# "Adaptive Spectral Defense for Real-Time Recovery of
#  Deep-Learning-Based Modulation Classifiers Under Adversarial Attack"
#
# Prerequisites:
#   - Python 3.8+ with PyTorch, torchattacks, numpy, pandas, matplotlib, seaborn, scipy
#   - RML2016.10a dataset at ./data/RML2016.10a_dict.pkl
#   - Pretrained AWN checkpoint at ./checkpoint/2016.10a_AWN.pkl
#   - TeX Live with latexmk, pdflatex, bibtex, and IEEEtran style
#
# Usage:
#   cd <repo_root>
#   bash paper/reproduce.sh           # full pipeline
#   bash paper/reproduce.sh --figures # figures + LaTeX only (skip expensive evaluation)
#
# Estimated runtime:
#   Full pipeline: ~4 hours on a single GPU (defense comparison is the bottleneck)
#   Figures only:  ~2 minutes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CKPT="./checkpoint"
DATASET="2016.10a"
RESULT_DIR="inference/2016.10a_165/result/defense_compare"
FIGURES_DIR="paper/latex/figures"

# Parse optional --figures flag
FIGURES_ONLY=false
for arg in "$@"; do
  if [[ "$arg" == "--figures" ]]; then
    FIGURES_ONLY=true
  fi
done

echo "=============================================="
echo "  Reproduce: Adaptive Spectral Defense Paper"
echo "=============================================="
echo ""

# ── Preflight checks ──────────────────────────────────────────────────────────

echo "=== Preflight: checking prerequisites ==="

if [[ ! -f "./data/RML2016.10a_dict.pkl" ]]; then
  echo "ERROR: dataset not found at ./data/RML2016.10a_dict.pkl"
  echo "  Download from: https://www.deepsig.ai/datasets"
  exit 1
fi

if [[ ! -f "$CKPT/2016.10a_AWN.pkl" ]]; then
  echo "ERROR: AWN checkpoint not found at $CKPT/2016.10a_AWN.pkl"
  echo "  Train the model first: python main.py --mode train --dataset 2016.10a"
  exit 1
fi

if ! python -c "import torch, torchattacks, pandas, matplotlib, seaborn, scipy" 2>/dev/null; then
  echo "ERROR: required Python packages missing."
  echo "  Install with: pip install -r requirements.txt"
  exit 1
fi

echo "  -> Prerequisites OK"
echo ""

# ── Optional: skip to figures ─────────────────────────────────────────────────

if [[ "$FIGURES_ONLY" == "true" ]]; then
  echo "  [--figures] Skipping evaluation steps; using existing CSVs in $RESULT_DIR"
  echo ""
else

  # ── Step 1: Calibrate classical filter parameters per-SNR ─────────────────
  echo "=== Step 1/4: Calibrate classical filter parameters (validation set) ==="
  echo "  Runs grid search over Kalman, Wiener, Savitzky-Golay, Gaussian, FIR params"
  echo "  per SNR level; result stored in calibration_params.json"
  python main.py \
    --mode calibrate_defenses \
    --dataset "$DATASET" \
    --ckpt_path "$CKPT"
  echo "  -> Calibration params saved to $RESULT_DIR/../calibration_params.json"
  echo ""

  # ── Step 2: Run full defense comparison evaluation ─────────────────────────
  echo "=== Step 2/4: Run defense comparison (9 defenses x 5 attacks x 10 SNRs) ==="
  echo "  Evaluates all defenses on CW, EAD-L1, EAD-EN, FGSM, PGD attacks."
  echo "  Also generates confusion matrices and perturbation budget curves."
  echo "  WARNING: CW and EAD optimization attacks take ~30-60 min each on GPU."
  python main.py \
    --mode defense_compare \
    --dataset "$DATASET" \
    --ckpt_path "$CKPT" \
    --max_per_cell 200 \
    --ta_box minmax \
    --attack_eps 0.03 \
    --cw_c 1.0 \
    --cw_steps 100 \
    --ead_initial_const 0.01 \
    --ead_beta 0.01 \
    --ead_steps 100
  echo "  -> Results saved to $RESULT_DIR/defense_compare.csv"
  echo "  -> Per-attack pivot tables: defense_compare_{cw,eadl1,eaden,fgsm,pgd}.csv"
  echo "  -> Confusion matrices: $RESULT_DIR/confmat/"
  echo "  -> Budget curves: $RESULT_DIR/budget_curves/"
  echo ""

  # ── Step 3: Generate frequency spectra figure (requires running environment)
  echo "=== Step 3/4: Generate frequency spectra (CW attack spectra) ==="
  echo "  Runs CW on a few QPSK samples to produce clean/attacked/recovered spectra."
  python main.py \
    --mode adv_eval \
    --dataset "$DATASET" \
    --ckpt_path "$CKPT" \
    --attack cw \
    --defense adaptive_k \
    --cw_c 1.0 \
    --cw_steps 100 \
    --mod_filter QPSK \
    --snr_filter 18
  echo "  -> Spectra data available for figure generation"
  echo ""

fi  # end if not FIGURES_ONLY

# ── Step 4: Generate all paper figures from CSVs/NPYs ─────────────────────────
echo "=== Step $([ "$FIGURES_ONLY" == "true" ] && echo "2" || echo "4")/$([ "$FIGURES_ONLY" == "true" ] && echo "2" || echo "5"): Generate paper figures from CSVs ==="
python paper/scripts/generate_figures.py
echo "  -> Figures saved to $FIGURES_DIR/"
echo "  -> Generated: defense_compare_overview.pdf, acc_vs_snr_*.pdf,"
echo "     confmat_*.pdf, budget_*.pdf, freq_spectra_cw.pdf"
echo ""

# ── Step 5: Compile LaTeX paper ────────────────────────────────────────────────
echo "=== Step $([ "$FIGURES_ONLY" == "true" ] && echo "3" || echo "5")/$([ "$FIGURES_ONLY" == "true" ] && echo "3" || echo "5"): Compile LaTeX paper ==="

if ! command -v latexmk &>/dev/null; then
  echo "WARNING: latexmk not found. Install TeX Live:"
  echo "  sudo apt-get install texlive-full"
  echo "  or: brew install mactex"
  exit 1
fi

cd paper/latex
latexmk -pdf -interaction=nonstopmode main.tex
echo "  -> Paper compiled: paper/latex/main.pdf"
PAGE_COUNT=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}' || echo "unknown")
echo "  -> Page count: $PAGE_COUNT"
cd "$REPO_ROOT"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "=============================================="
echo "  Reproduction complete!"
echo ""
echo "  Paper PDF:  paper/latex/main.pdf  ($PAGE_COUNT pages)"
echo "  Results:    $RESULT_DIR/"
echo "  Figures:    $FIGURES_DIR/"
echo ""
echo "  Key result files:"
echo "    $RESULT_DIR/defense_compare.csv         (all 9 defenses x 5 attacks)"
echo "    $RESULT_DIR/defense_compare_cw.csv      (CW pivot table by SNR)"
echo "    $RESULT_DIR/confmat/confmat_summary.csv (before/after accuracy)"
echo "    $RESULT_DIR/budget_curves/budget_fgsm.csv"
echo "    $RESULT_DIR/budget_curves/budget_cw.csv"
echo "=============================================="
