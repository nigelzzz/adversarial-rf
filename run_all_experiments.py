#!/usr/bin/env python3
"""
Run all Phase 2 experiments: defense comparison + confusion matrices + budget curves.
Uses adaptive_k defense (replaces ae_fft_topk).
Reuses existing inference/2016.10a_165 directory.

Run: python3 run_all_experiments.py
"""
import os
import sys
import glob
import torch
import logging

from util.config import Config, merge_args2cfg
from util.logger import create_logger
from data_loader.data_loader import Load_Dataset, Dataset_Split
from util.utils import create_AWN_model

# Setup
dataset = '2016.10a'
ckpt_path = './checkpoint'
max_per_cell = 200

# Build cfg — force reuse of existing inference dir
cfg = Config(dataset, train=False)
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.cfg_dir = 'inference/2016.10a_165'
cfg.model_dir = os.path.join(cfg.cfg_dir, 'models')
cfg.log_dir = os.path.join(cfg.cfg_dir, 'log')
cfg.result_dir = os.path.join(cfg.cfg_dir, 'result')
# Attack defaults (D-05: eps=0.03 minmax for Linf, c=1.0 for CW/EAD)
cfg.attack_eps = 0.03
cfg.cw_c = 1.0
cfg.cw_steps = 100
cfg.cw_lr = 0.001
cfg.cw_kappa = 1.0
cfg.ta_box = 'minmax'
cfg.defense = 'adaptive_k'
cfg.def_topk = 50
cfg.detector_threshold = 0.004468
cfg.ead_kappa = 1.0
cfg.ead_max_iterations = 200
cfg.ead_binary_search_steps = 9
cfg.ead_initial_const = 1.0

logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
logger.info("=" * 60)
logger.info("Phase 2: Full experiments with adaptive_k defense")
logger.info("=" * 60)

# Model
model = create_AWN_model(cfg)
model.load_state_dict(torch.load(
    os.path.join(ckpt_path, f'{dataset}_AWN.pkl'),
    map_location=cfg.device,
    weights_only=True,
))
model.eval()
logger.info("Model loaded on %s", cfg.device)

# Detector (still needed for spectral_gated if it uses it, pass None if not needed)
detector = None
try:
    from util.detector import RFSignalAutoEncoder
    det_path = './checkpoint/detector_ae.pth'
    if os.path.isfile(det_path):
        detector = RFSignalAutoEncoder().to(cfg.device)
        detector.load_state_dict(torch.load(det_path, map_location=cfg.device, weights_only=True))
        detector.eval()
        logger.info("Detector loaded")
except Exception as e:
    logger.info("Detector not loaded: %s", e)

# Data
Signals, Labels, SNRs, snrs, mods = Load_Dataset(dataset, logger)
train_set, test_set, val_set, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
Signals_test, Labels_test = test_set

# Auto-detect calibration params
candidates = sorted(glob.glob('inference/*/result/calibration_params.json'))
calibration_path = candidates[-1] if candidates else None
if calibration_path:
    logger.info("Using calibration params: %s", calibration_path)

# ── Stage 1: Defense comparison table ──────────────────────────────────────
logger.info("\n=== STAGE 1: Defense comparison table (9 defenses x 5 attacks x 10 SNRs) ===")
from util.defense_compare import run_defense_compare
run_defense_compare(
    model, Signals_test, Labels_test, SNRs, test_idx, cfg, logger,
    detector=detector,
    max_per_cell=max_per_cell,
    calibration_path=calibration_path,
)

# ── Stage 2: Confusion matrices ───────────────────────────────────────────
logger.info("\n=== STAGE 2: Confusion matrices (3 attacks x 3 SNRs x before/after) ===")
from util.defense_compare import generate_confusion_matrices
generate_confusion_matrices(
    model, Signals_test, Labels_test, SNRs, test_idx, cfg, logger,
    detector=detector,
    max_per_cell=max_per_cell,
    calibration_path=calibration_path,
)

# ── Stage 3: Budget curves ────────────────────────────────────────────────
logger.info("\n=== STAGE 3: Budget curves (Linf eps sweep + optimization c sweep) ===")
from util.defense_compare import generate_budget_curves
generate_budget_curves(
    model, Signals_test, Labels_test, SNRs, test_idx, cfg, logger,
    detector=detector,
    max_per_cell=max_per_cell,
    calibration_path=calibration_path,
)

logger.info("\n=== ALL EXPERIMENTS COMPLETE ===")
print("\nAll done! Results in: inference/2016.10a_165/result/defense_compare/")
