#!/usr/bin/env python3
"""
One-off script to generate only the budget curves (EVAL-04).
Reuses existing inference/2016.10a_165 directory.
Run: python run_budget_curves_only.py
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
detector_ckpt = './checkpoint/detector_ae.pth'
max_per_cell = 200

# Build cfg — force reuse of existing inference dir
cfg = Config(dataset, train=False)
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.cfg_dir = 'inference/2016.10a_165'
cfg.model_dir = os.path.join(cfg.cfg_dir, 'models')
cfg.log_dir = os.path.join(cfg.cfg_dir, 'log')
cfg.result_dir = os.path.join(cfg.cfg_dir, 'result')
# Set attack defaults needed by defense_compare
cfg.attack_eps = 0.03
cfg.cw_c = 1.0
cfg.cw_steps = 100
cfg.cw_lr = 0.01
cfg.cw_kappa = 0.0
cfg.ta_box = 'minmax'
cfg.defense = 'ae_fft_topk'
cfg.def_topk = 50
cfg.detector_threshold = 0.004468
cfg.ead_initial_const = 1.0

logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
logger.info("=== Budget Curves Only ===")

# Model
model = create_AWN_model(cfg)
model.load_state_dict(torch.load(
    os.path.join(ckpt_path, f'{dataset}_AWN.pkl'),
    map_location=cfg.device,
    weights_only=True,
))
model.eval()

# Detector
from util.detector import RFSignalAutoEncoder
detector = RFSignalAutoEncoder().to(cfg.device)
detector.load_state_dict(torch.load(detector_ckpt, map_location=cfg.device, weights_only=True))
detector.eval()

# Data
Signals, Labels, SNRs, snrs, mods = Load_Dataset(dataset, logger)
train_set, test_set, val_set, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
Signals_test, Labels_test = test_set

# Auto-detect calibration params
candidates = sorted(glob.glob('inference/*/result/calibration_params.json'))
calibration_path = candidates[-1] if candidates else None
if calibration_path:
    logger.info("Using calibration params: %s", calibration_path)

# Run budget curves
from util.defense_compare import generate_budget_curves
generate_budget_curves(
    model,
    Signals_test,
    Labels_test,
    SNRs,
    test_idx,
    cfg,
    logger,
    detector=detector,
    max_per_cell=max_per_cell,
    calibration_path=calibration_path,
)

print("\nDone! Check: inference/2016.10a_165/result/defense_compare/budget_curves/")
