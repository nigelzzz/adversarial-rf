import os
import random
import sys

import numpy as np
import torch

from models.model import AWN
from models.vtcnn2 import VTCNN2
from models.resnet1d import ResNet1D
from models.lstm_amc import LSTM_AMC
# Lazy import MCLDNN (it's a git submodule that may not be cloned)
_MCLDNN_PyTorch = None

def _get_mcldnn():
    global _MCLDNN_PyTorch
    if _MCLDNN_PyTorch is None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MCLDNN'))
        from mcldnn_pytorch import MCLDNN_PyTorch
        _MCLDNN_PyTorch = MCLDNN_PyTorch
    return _MCLDNN_PyTorch


def rrc_filter(beta, sps, num_taps=101):
    """Root Raised Cosine filter."""
    T = 1.0  # symbol period
    t = np.arange(-(num_taps // 2), num_taps // 2 + 1) / sps
    h = np.zeros_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = (1 + beta * (4 / np.pi - 1))
        elif abs(abs(ti) - T / (4 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


def recover_constellation(I, Q, sps=8, beta=0.35, mod_order=4):
    """
    Recover constellation points from raw oversampled IQ data.

    Applies matched filter (RRC), symbol-rate downsampling, and
    blind phase recovery (Viterbi & Viterbi for M-th power).

    Args:
        I: In-phase samples, 1-D numpy array
        Q: Quadrature samples, 1-D numpy array
        sps: Samples per symbol (8 for RML2016.10a)
        beta: Roll-off factor for RRC filter
        mod_order: Modulation order for phase recovery
                   (2=BPSK, 4=QPSK/QAM, use 4 as safe default)

    Returns:
        I_sym, Q_sym: Symbol-rate constellation points
    """
    # Build complex signal
    s = I + 1j * Q

    # 1. Matched filter (RRC)
    h = rrc_filter(beta, sps, num_taps=min(8 * sps + 1, len(s) // 2))
    y = np.convolve(s, h, mode='same')

    # 2. Symbol-rate downsample
    # Try multiple offsets and pick the one with max eye opening (power variance)
    best_offset = 0
    best_var = -1
    for offset in range(sps):
        syms = y[offset::sps]
        v = np.var(np.abs(syms))
        if v > best_var:
            best_var = v
            best_offset = offset
    sym = y[best_offset::sps]

    if len(sym) < 2:
        return I[::sps], Q[::sps]

    # 3. Blind phase recovery (M-th power Viterbi & Viterbi)
    M = mod_order
    phase_est = np.angle(np.mean(sym ** M)) / M
    sym = sym * np.exp(-1j * phase_est)

    return sym.real, sym.imag


def fix_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_exp_settings(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 20)
    log_dict = cfg.__dict__.copy()
    for k, v in log_dict.items():
        logger.info(f'{k} : {v}')
    logger.info('=' * 20)


def create_AWN_model(cfg):
    """
    build AWN model
    """
    model = AWN(
        num_classes=cfg.num_classes,
        num_levels=cfg.num_level,
        in_channels=cfg.in_channels,
        kernel_size=cfg.kernel_size,
        latent_dim=cfg.latent_dim,
        regu_details=cfg.regu_details,
        regu_approx=cfg.regu_approx,
    ).to(cfg.device)

    return model


def create_MCLDNN_model(cfg):
    """
    build MCLDNN model (PyTorch version)
    """
    MCLDNNCls = _get_mcldnn()
    model = MCLDNNCls(
        num_classes=cfg.num_classes,
        dropout_rate=0.5,
    ).to(cfg.device)

    return model


def create_VTCNN2_model(cfg):
    """
    build VTCNN2 model
    """
    signal_len = getattr(cfg, 'signal_len', 128)
    model = VTCNN2(
        num_classes=cfg.num_classes,
        dropout_rate=0.5,
        signal_len=signal_len,
    ).to(cfg.device)
    return model


def create_ResNet1D_model(cfg):
    """
    build ResNet1D model
    """
    model = ResNet1D(
        num_classes=cfg.num_classes,
    ).to(cfg.device)
    return model


def create_LSTM_model(cfg):
    """
    build LSTM_AMC model
    """
    model = LSTM_AMC(
        num_classes=cfg.num_classes,
    ).to(cfg.device)
    return model


def create_model(cfg, model_name='awn'):
    """
    Factory function to create model by name.

    Args:
        cfg: Config object
        model_name: 'awn', 'mcldnn', 'vtcnn2', 'resnet1d', or 'lstm'

    Returns:
        model: PyTorch model
    """
    model_name = model_name.lower()
    if model_name == 'awn':
        return create_AWN_model(cfg)
    elif model_name == 'mcldnn':
        return create_MCLDNN_model(cfg)
    elif model_name == 'vtcnn2':
        return create_VTCNN2_model(cfg)
    elif model_name == 'resnet1d':
        return create_ResNet1D_model(cfg)
    elif model_name == 'lstm':
        return create_LSTM_model(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose 'awn', 'mcldnn', 'vtcnn2', 'resnet1d', or 'lstm'.")
