import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from util.defense import fft_topk_percent_denoise


def _predict_labels(
    model: torch.nn.Module,
    x: torch.Tensor,
    cfg,
    *,
    percent: float = None,
) -> torch.Tensor:
    """
    Run model prediction on batches, optionally keeping only the top-k% FFT bins.
    """
    preds: List[torch.Tensor] = []
    for batch in torch.chunk(x, cfg.test_batch_size, dim=0):
        batch = batch.to(cfg.device)
        if percent is not None:
            batch = fft_topk_percent_denoise(batch, percent=float(percent))
        with torch.no_grad():
            logits, _ = model(batch)
        preds.append(torch.argmax(logits, dim=1).cpu())
    return torch.cat(preds, dim=0)


def _acc_by_mod(
    preds: np.ndarray,
    labels: np.ndarray,
    classes: Dict[bytes, int],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute overall accuracy and per-modulation accuracy.
    """
    overall = float(np.mean(preds == labels))
    per_mod: Dict[str, float] = {}
    for mod_name, mod_idx in classes.items():
        mask = labels == mod_idx
        if mask.sum() == 0:
            continue
        key = mod_name.decode() if isinstance(mod_name, bytes) else str(mod_name)
        per_mod[key] = float(np.mean(preds[mask] == labels[mask]))
    return overall, per_mod


def run_freq_topk_eval(
    model: torch.nn.Module,
    sig_test: torch.Tensor,
    lab_test: torch.Tensor,
    SNRs,
    test_idx,
    cfg,
    logger,
    *,
    snr_min: float = 0.0,
    percents: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    eval_limit: int = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate accuracy on clean signals and after keeping only the top-k% FFT magnitudes.

    Args:
        model: Trained AWN model.
        sig_test: Test signals tensor [N, 2, T].
        lab_test: Test labels tensor [N].
        SNRs: Full SNR list aligned with dataset indices.
        test_idx: Indices of test samples within the full dataset.
        cfg: Config object (for device and batch size).
        logger: Logger to record progress.
        snr_min: Keep only samples with SNR >= snr_min.
        percents: Iterable of percentages (0..1] to keep in FFT.
        eval_limit: Optional cap on number of filtered samples for speed.

    Returns:
        Nested dict of accuracies keyed by variant name.
    """
    model.eval()

    test_snrs = np.array([SNRs[i] for i in test_idx])
    mask = test_snrs >= snr_min
    if not mask.any():
        raise SystemExit(f'No test samples satisfy SNR >= {snr_min}')

    sig_sel = sig_test[mask]
    lab_sel = lab_test[mask]
    if eval_limit is not None:
        sig_sel = sig_sel[:eval_limit]
        lab_sel = lab_sel[:eval_limit]
        logger.info(f'Limiting eval to first {sig_sel.shape[0]} samples after SNR filter.')

    logger.info(
        f'Filtered test set: {sig_sel.shape[0]} samples, '
        f'SNR >= {snr_min}, mods={len(cfg.classes)}'
    )

    results: Dict[str, Dict[str, float]] = {}

    preds_clean = _predict_labels(model, sig_sel, cfg, percent=None)
    overall, per_mod = _acc_by_mod(preds_clean.numpy(), lab_sel.numpy(), cfg.classes)
    results['clean'] = {'overall': overall, **per_mod}
    logger.info(f'Clean overall accuracy: {overall:.4f}')

    for pct in percents:
        name = f'top{int(float(pct) * 100)}'
        preds = _predict_labels(model, sig_sel, cfg, percent=pct)
        overall_pct, per_mod_pct = _acc_by_mod(preds.numpy(), lab_sel.numpy(), cfg.classes)
        results[name] = {'overall': overall_pct, **per_mod_pct}
        per_mod_str = ', '.join([f'{k}={v:.4f}' for k, v in sorted(per_mod_pct.items())])
        logger.info(f'{name} accuracy: overall={overall_pct:.4f} | {per_mod_str}')

    out_dir = os.path.join(cfg.result_dir, 'freq_topk')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'freq_topk_eval.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Saved results to {out_path}')

    return results
