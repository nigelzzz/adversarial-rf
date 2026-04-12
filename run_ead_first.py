#!/usr/bin/env python3
"""Run only EADL1 + EADEN with adaptive_k defense and updated params."""
import os, glob, torch
from util.config import Config
from util.logger import create_logger
from data_loader.data_loader import Load_Dataset, Dataset_Split
from util.utils import create_AWN_model

dataset = '2016.10a'
cfg = Config(dataset, train=False)
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.cfg_dir = 'inference/2016.10a_165'
cfg.model_dir = os.path.join(cfg.cfg_dir, 'models')
cfg.log_dir = os.path.join(cfg.cfg_dir, 'log')
cfg.result_dir = os.path.join(cfg.cfg_dir, 'result')
# Attack params (updated)
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
logger.info("EAD-only run: adaptive_k, kappa=1, const=10, iter=200")
logger.info("=" * 60)

model = create_AWN_model(cfg)
model.load_state_dict(torch.load(
    os.path.join('./checkpoint', f'{dataset}_AWN.pkl'),
    map_location=cfg.device, weights_only=True))
model.eval()

detector = None
try:
    from util.detector import RFSignalAutoEncoder
    det_path = './checkpoint/detector_ae.pth'
    if os.path.isfile(det_path):
        detector = RFSignalAutoEncoder().to(cfg.device)
        detector.load_state_dict(torch.load(det_path, map_location=cfg.device, weights_only=True))
        detector.eval()
except: pass

Signals, Labels, SNRs, snrs, mods = Load_Dataset(dataset, logger)
_, test_set, _, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
Signals_test, Labels_test = test_set

candidates = sorted(glob.glob('inference/*/result/calibration_params.json'))
calibration_path = candidates[-1] if candidates else None
if calibration_path:
    logger.info("Calibration: %s", calibration_path)

from util.defense_compare import run_defense_compare
run_defense_compare(
    model, Signals_test, Labels_test, SNRs, test_idx, cfg, logger,
    detector=detector,
    attacks=['eadl1', 'eaden'],
    max_per_cell=200,
    calibration_path=calibration_path,
)
print("\nDone! Check: inference/2016.10a_165/result/defense_compare/")
