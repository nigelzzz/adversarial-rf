import os
import json
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from util.adv_attack import cw_l2_attack, spectral_noise_attack, _batch_clip


def _complex_rfft_psd(x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average one-sided PSD of complex baseband x (I/Q) over batch.

    Args:
        x: Tensor [N, 2, T] (I, Q)

    Returns:
        f: numpy array of normalized frequency bins in [0, 0.5], length F=T//2+1
        psd_db: numpy array of mean power spectral density in dB, length F
    """
    assert x.dim() == 3 and x.size(1) == 2
    N, C, T = x.shape
    device = x.device
    dtype = x.dtype

    # Build complex baseband
    z = x[:, 0, :] + 1j * x[:, 1, :]
    Z = torch.fft.fft(z, n=T, dim=1)
    Zp = Z[:, : T // 2 + 1]
    # Power spectral density (simple periodogram mean over batch)
    psd = (Zp.abs() ** 2) / T  # [N, F]
    psd_mean = psd.mean(dim=0)
    psd_db = 10.0 * torch.log10(psd_mean.clamp_min(1e-12))
    F = Zp.shape[1]
    f = np.linspace(0.0, 0.5, num=F, endpoint=True)
    return f, psd_db.detach().cpu().numpy()


def run_freq_compare(model,
                     sig_test: torch.Tensor,
                     lab_test: torch.Tensor,
                     SNRs,
                     test_idx,
                     cfg,
                     logger,
                     *,
                     tone_freq: Optional[float] = None,
                     spec_eps: float = 0.1,
                     spec_type: str = 'cw_tone',
                     spec_mask_path: Optional[str] = None,
                     spec_band_low: float = 0.05,
                     spec_band_high: float = 0.25) -> None:
    """
    Generate two frequency-domain comparisons (PSD overlays):
      1) Original vs CW adversarial
      2) Original vs CW tone (spectral) adversarial

    Saves two .svg files under cfg.result_dir/freq/.
    """
    model.eval()

    # Filter to the requested SNR/mod via cfg (already supported by Load_Dataset in main)
    # Build test tensors for this SNR only (Dataset_Split already applied filters)
    snrs = list(np.unique(SNRs))
    assert len(snrs) == 1, "Please pass a single SNR via --snr_filter"

    # Optional limit for speed
    limit = getattr(cfg, 'eval_limit', None)

    # Split set into chunks
    Sample = torch.chunk(sig_test[:limit] if limit is not None else sig_test, cfg.test_batch_size, dim=0)
    Label = torch.chunk(lab_test[:limit] if limit is not None else lab_test, cfg.test_batch_size, dim=0)

    # Optional mask load
    mask_t = None
    if spec_type in ('psd_mask', 'mask') and spec_mask_path is not None:
        try:
            mask_np = np.load(spec_mask_path)
            mask_t = torch.as_tensor(mask_np)
            logger.info(f'Loaded PSD mask: {spec_mask_path} (len={len(mask_np)})')
        except Exception as e:
            logger.info(f'Failed to load PSD mask {spec_mask_path}: {e}')

    # Collect originals for PSD
    x_list = []
    x_cw_list = []
    x_tone_list = []

    for (sample, label) in zip(Sample, Label):
        sample = sample.to(cfg.device)
        label = label.to(cfg.device)
        x_list.append(sample.cpu())

        # CW adversarial (reuse cw_* args from cfg)
        adv_cw = cw_l2_attack(
            model,
            sample,
            y=label,
            targeted=getattr(cfg, 'cw_targeted', False),
            c=getattr(cfg, 'cw_c', 0.5),
            kappa=getattr(cfg, 'cw_kappa', 0.0),
            steps=getattr(cfg, 'cw_steps', 10),
            lr=getattr(cfg, 'cw_lr', 1e-2),
            lowpass=getattr(cfg, 'lowpass', True),
            lowpass_kernel=getattr(cfg, 'lowpass_kernel', 17),
            device=cfg.device,
        )
        x_cw_list.append(adv_cw.cpu())

        # Spectral perturbation (tone/band/mask)
        kwargs = dict(spec_type=spec_type, spec_eps=spec_eps, tone_freq=tone_freq)
        if spec_type in ('psd_band', 'band'):
            kwargs['band'] = (spec_band_low, spec_band_high)
        if spec_type in ('psd_mask', 'mask'):
            kwargs['psd_mask'] = mask_t
        adv_tone = spectral_noise_attack(sample, **kwargs)
        x_tone_list.append(adv_tone.cpu())

    x = torch.cat(x_list, dim=0)
    x_cw = torch.cat(x_cw_list, dim=0)
    x_tone = torch.cat(x_tone_list, dim=0)

    f, psd_orig = _complex_rfft_psd(x)
    _, psd_cw = _complex_rfft_psd(x_cw)
    _, psd_tone = _complex_rfft_psd(x_tone)

    out_dir = os.path.join(cfg.result_dir, 'freq')
    os.makedirs(out_dir, exist_ok=True)

    # Plot CW vs Orig
    plt.figure(figsize=(7, 4))
    plt.plot(f, psd_orig, label='Original')
    plt.plot(f, psd_cw, label='CW adversarial')
    plt.xlabel('Normalized frequency [cycles/sample]')
    plt.ylabel('PSD [dB]')
    plt.title('PSD: Original vs CW (avg over batch)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    cw_path = os.path.join(out_dir, 'psd_compare_cw.svg')
    plt.savefig(cw_path, format='svg', dpi=150)
    plt.close()
    logger.info(f'Saved {cw_path}')

    # Plot Spectral vs Orig
    plt.figure(figsize=(7, 4))
    plt.plot(f, psd_orig, label='Original')
    label = 'Spectral adversarial'
    if spec_type == 'cw_tone':
        label = 'CW tone adversarial'
    elif spec_type in ('psd_band','band'):
        label = 'Band-limited adversarial'
    elif spec_type in ('psd_mask','mask'):
        label = 'PSD-mask adversarial'
    plt.plot(f, psd_tone, label=label)
    plt.xlabel('Normalized frequency [cycles/sample]')
    plt.ylabel('PSD [dB]')
    plt.title('PSD: Original vs Spectral (avg over batch)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    tone_path = os.path.join(out_dir, 'psd_compare_spectral.svg')
    plt.savefig(tone_path, format='svg', dpi=150)
    plt.close()
    logger.info(f'Saved {tone_path}')

    # Closeness metrics between CW and spectral (MAE in dB)
    mae_db = float(np.mean(np.abs(psd_cw - psd_tone)))
    rmse_db = float(np.sqrt(np.mean((psd_cw - psd_tone) ** 2)))
    logger.info(f'PSD closeness (CW vs Spectral): MAE={mae_db:.3f} dB, RMSE={rmse_db:.3f} dB')

    # MSE on linear power scale between pairs
    psd_orig_lin = np.power(10.0, psd_orig / 10.0)
    psd_cw_lin = np.power(10.0, psd_cw / 10.0)
    psd_spec_lin = np.power(10.0, psd_tone / 10.0)
    mse_cw_vs_orig = float(np.mean((psd_cw_lin - psd_orig_lin) ** 2))
    mse_spec_vs_orig = float(np.mean((psd_spec_lin - psd_orig_lin) ** 2))
    mse_cw_vs_spec = float(np.mean((psd_cw_lin - psd_spec_lin) ** 2))
    logger.info(f'MSE (linear PSD): CW vs Orig={mse_cw_vs_orig:.6f}, Spectral vs Orig={mse_spec_vs_orig:.6f}, CW vs Spectral={mse_cw_vs_spec:.6f}')

    # If band-limited, also report in-band/out-of-band MSEs
    inband = outband = None
    if spec_type in ('psd_band', 'band'):
        F = psd_orig.shape[0]
        fvec = np.linspace(0.0, 0.5, num=F, endpoint=True)
        mask_band = (fvec >= spec_band_low) & (fvec <= spec_band_high)
        mask_out = ~mask_band
        if mask_band.any() and mask_out.any():
            inband = {
                'mse_cw_vs_orig': float(np.mean((psd_cw_lin[mask_band] - psd_orig_lin[mask_band]) ** 2)),
                'mse_spec_vs_orig': float(np.mean((psd_spec_lin[mask_band] - psd_orig_lin[mask_band]) ** 2)),
                'mse_cw_vs_spec': float(np.mean((psd_cw_lin[mask_band] - psd_spec_lin[mask_band]) ** 2)),
            }
            outband = {
                'mse_cw_vs_orig': float(np.mean((psd_cw_lin[mask_out] - psd_orig_lin[mask_out]) ** 2)),
                'mse_spec_vs_orig': float(np.mean((psd_spec_lin[mask_out] - psd_orig_lin[mask_out]) ** 2)),
                'mse_cw_vs_spec': float(np.mean((psd_cw_lin[mask_out] - psd_spec_lin[mask_out]) ** 2)),
            }
            logger.info(
                'In-band MSE (linear PSD): CW vs Orig={:.6f}, Spectral vs Orig={:.6f}, CW vs Spectral={:.6f}'.format(
                    inband['mse_cw_vs_orig'], inband['mse_spec_vs_orig'], inband['mse_cw_vs_spec']))
            logger.info(
                'Out-of-band MSE (linear PSD): CW vs Orig={:.6f}, Spectral vs Orig={:.6f}, CW vs Spectral={:.6f}'.format(
                    outband['mse_cw_vs_orig'], outband['mse_spec_vs_orig'], outband['mse_cw_vs_spec']))

    metrics = {
        'mae_db': mae_db,
        'rmse_db': rmse_db,
        'spec_type': spec_type,
        'eval_limit': getattr(cfg, 'eval_limit', None),
        'cw_c': getattr(cfg, 'cw_c', None),
        'cw_steps': getattr(cfg, 'cw_steps', None),
        'lowpass': getattr(cfg, 'lowpass', None),
        'mse_cw_vs_orig': mse_cw_vs_orig,
        'mse_spec_vs_orig': mse_spec_vs_orig,
        'mse_cw_vs_spec': mse_cw_vs_spec,
    }
    if inband is not None and outband is not None:
        metrics['inband_mse'] = inband
        metrics['outband_mse'] = outband
    metrics_path = os.path.join(out_dir, 'psd_compare_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'Saved {metrics_path}')
