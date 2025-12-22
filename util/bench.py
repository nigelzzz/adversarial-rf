import os
import json
import time
from typing import Optional

import numpy as np
import torch

from util.adv_attack import cw_l2_attack, spectral_noise_attack


@torch.no_grad()
def _make_spectral_kwargs(cfg, mask_loader=None):
    spec_type = getattr(cfg, 'spec_type', 'cw_tone')
    kwargs = dict(spec_type=spec_type,
                  spec_eps=getattr(cfg, 'spec_eps', 0.1),
                  jnr_db=getattr(cfg, 'spec_jnr_db', None),
                  tone_freq=getattr(cfg, 'tone_freq', None))
    if spec_type in ('psd_band', 'band'):
        kwargs['band'] = (
            getattr(cfg, 'spec_band_low', 0.05),
            getattr(cfg, 'spec_band_high', 0.25),
        )
    if spec_type in ('psd_mask', 'mask'):
        if mask_loader is None:
            mask_path = getattr(cfg, 'spec_mask_path', None)
            if mask_path is not None:
                try:
                    import numpy as _np
                    mask_np = _np.load(mask_path)
                    mask_t = torch.as_tensor(mask_np)
                except Exception:
                    mask_t = None
            else:
                mask_t = None
        else:
            mask_t = mask_loader()
        kwargs['psd_mask'] = mask_t
    return kwargs


def run_attack_bench(model,
                     sig_test: torch.Tensor,
                     lab_test: torch.Tensor,
                     cfg,
                     logger) -> None:
    """
    Benchmark adversarial generation latency for CW optimizer vs Spectral PSD mask.

    Uses cfg.test_batch_size chunks and cfg.eval_limit examples (if set).
    Saves metrics JSON under cfg.result_dir/bench/adv_bench.json.
    """
    model.eval()

    limit = getattr(cfg, 'eval_limit', None)
    x_all = sig_test[:limit] if limit is not None else sig_test
    y_all = lab_test[:limit] if limit is not None else lab_test

    # Warmup: a small forward for model (does not affect spectral timing much)
    with torch.no_grad():
        _ = model(x_all[: min(4, x_all.size(0))].to(cfg.device))

    # Chunking
    Sample = torch.chunk(x_all, cfg.test_batch_size, dim=0)
    Label = torch.chunk(y_all, cfg.test_batch_size, dim=0)

    # Measure CW
    t0 = time.perf_counter()
    for (sample, label) in zip(Sample, Label):
        sample = sample.to(cfg.device)
        label = label.to(cfg.device)
        _ = cw_l2_attack(
            model,
            sample,
            y=label,
            targeted=getattr(cfg, 'cw_targeted', False),
            c=getattr(cfg, 'cw_c', 1.0),
            kappa=getattr(cfg, 'cw_kappa', 0.0),
            steps=getattr(cfg, 'cw_steps', 100),
            lr=getattr(cfg, 'cw_lr', 1e-2),
            lowpass=getattr(cfg, 'lowpass', True),
            lowpass_kernel=getattr(cfg, 'lowpass_kernel', 17),
            device=cfg.device,
        )
    cw_time = time.perf_counter() - t0

    # Measure Spectral
    spec_kwargs = _make_spectral_kwargs(cfg)
    t0 = time.perf_counter()
    for (sample, label) in zip(Sample, Label):
        sample = sample.to(cfg.device)
        _ = spectral_noise_attack(sample, **spec_kwargs)
    spec_time = time.perf_counter() - t0

    n_samples = x_all.size(0)
    metrics = {
        'n_samples': int(n_samples),
        'batch_size': int(cfg.test_batch_size),
        'device': str(cfg.device),
        'cw_steps': int(getattr(cfg, 'cw_steps', 100)),
        'cw_c': float(getattr(cfg, 'cw_c', 1.0)),
        'lowpass': bool(getattr(cfg, 'lowpass', True)),
        'spec_type': getattr(cfg, 'spec_type', 'cw_tone'),
        'spec_eps': float(getattr(cfg, 'spec_eps', 0.1)),
        'spec_jnr_db': getattr(cfg, 'spec_jnr_db', None),
        'tone_freq': getattr(cfg, 'tone_freq', None),
        'total_time_cw_s': cw_time,
        'total_time_spec_s': spec_time,
        'per_sample_cw_ms': (cw_time / n_samples) * 1000.0,
        'per_sample_spec_ms': (spec_time / n_samples) * 1000.0,
        'speedup_spec_vs_cw': cw_time / spec_time if spec_time > 0 else float('inf'),
    }

    bench_dir = os.path.join(cfg.result_dir, 'bench')
    os.makedirs(bench_dir, exist_ok=True)
    out_path = os.path.join(bench_dir, 'adv_bench.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"Bench (n={n_samples}, bs={cfg.test_batch_size}) | CW: {metrics['per_sample_cw_ms']:.2f} ms/sample, "
        f"Spectral: {metrics['per_sample_spec_ms']:.2f} ms/sample, speedup x{metrics['speedup_spec_vs_cw']:.1f}")
    logger.info(f"Saved {out_path}")

