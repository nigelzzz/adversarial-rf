import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from util.adv_attack import (
    Model01Wrapper,
    _batch_clip,
    _lowpass_filter,
    cw_l2_attack,
    iq_to_ta_input,
    iq_to_ta_input_minmax,
    ta_output_to_iq,
    ta_output_to_iq_minmax,
    spectral_noise_attack,
)
from util.defense import fft_topk_percent_denoise
from util.freq_topk_eval import _acc_by_mod


def _predict_labels(
    model: torch.nn.Module,
    x: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """
    Run model prediction on batches.
    """
    preds: List[torch.Tensor] = []
    for batch in torch.chunk(x, cfg.test_batch_size, dim=0):
        batch = batch.to(cfg.device)
        with torch.no_grad():
            logits, _ = model(batch)
        preds.append(torch.argmax(logits, dim=1).cpu())
    return torch.cat(preds, dim=0)


def _cw_attack_batch(
    model: torch.nn.Module,
    sample: torch.Tensor,
    label: torch.Tensor,
    cfg,
    logger,
) -> torch.Tensor:
    """
    Generate CW adversarial examples for a batch, mirroring Run_Adv_Eval behavior.
    """
    backend = getattr(cfg, 'attack_backend', 'torchattacks')
    if backend == 'torchattacks':
        try:
            import torchattacks

            wrapped_model = Model01Wrapper(model).eval()
            atk = torchattacks.CW(
                wrapped_model,
                c=getattr(cfg, 'cw_c', 1.0),
                kappa=getattr(cfg, 'cw_kappa', 0.0),
                steps=getattr(cfg, 'cw_steps', 100),
                lr=getattr(cfg, 'cw_lr', 1e-2),
            )
            box = str(getattr(cfg, 'ta_box', 'unit')).lower()
            if box == 'minmax':
                x01_4d, a, b = iq_to_ta_input_minmax(sample)
                wrapped_model.set_minmax(a, b)
                adv01_4d = atk(x01_4d, label)
                adv = ta_output_to_iq_minmax(adv01_4d, a, b)
                wrapped_model.clear_minmax()
            else:
                x01_4d = iq_to_ta_input(sample)
                adv01_4d = atk(x01_4d, label)
                adv = ta_output_to_iq(adv01_4d)
            if getattr(cfg, 'lowpass', True):
                delta = adv - sample
                delta = _lowpass_filter(delta, kernel_size=getattr(cfg, 'lowpass_kernel', 17))
                clip_min = sample.amin(dim=(1, 2), keepdim=True)
                clip_max = sample.amax(dim=(1, 2), keepdim=True)
                adv = _batch_clip(sample + delta, clip_min, clip_max)
            cw_scale = getattr(cfg, 'cw_scale', None)
            if cw_scale is not None:
                try:
                    s = float(cw_scale)
                    if s < 1.0:
                        delta = adv - sample
                        adv = _batch_clip(
                            sample + s * delta,
                            sample.amin(dim=(1, 2), keepdim=True),
                            sample.amax(dim=(1, 2), keepdim=True),
                        )
                except Exception:
                    pass
            return adv
        except Exception as e:
            logger.info(f"Falling back to internal CW due to: {e}")

    # Internal CW fallback
    adv = cw_l2_attack(
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
    return adv


def _spectral_attack_batch(
    sample: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """
    Add non-optimized spectral noise (e.g., low-frequency band) using spectral_noise_attack.
    """
    eps = getattr(cfg, 'spec_eps', 0.1)
    if eps is not None and eps < 0:
        eps = None
    return spectral_noise_attack(
        sample,
        spec_type=getattr(cfg, 'spec_type', 'psd_band'),
        spec_eps=eps,
        jnr_db=getattr(cfg, 'spec_jnr_db', None),
        tone_freq=getattr(cfg, 'tone_freq', None),
        band=(getattr(cfg, 'spec_band_low', 0.0), getattr(cfg, 'spec_band_high', 0.1)),
        psd_mask=None,
    )


def run_freq_topk_adv_eval(
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
    Evaluate clean vs CW adversarial accuracy and recovery via FFT top-k% keeping.

    Args:
        model: Trained AWN model.
        sig_test: Test signals tensor [N, 2, T].
        lab_test: Test labels tensor [N].
        SNRs: Full SNR list aligned with dataset indices.
        test_idx: Indices of test samples within the full dataset.
        cfg: Config object (device/batch size/hparams).
        logger: Logger.
        snr_min: Keep samples with SNR >= snr_min.
        percents: Percents (0..1] to keep in FFT for recovery.
        eval_limit: Optional cap on number of samples after SNR filter.
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

    preds_clean = _predict_labels(model, sig_sel, cfg)
    overall, per_mod = _acc_by_mod(preds_clean.numpy(), lab_sel.numpy(), cfg.classes)
    results['clean'] = {'overall': overall, **per_mod}
    logger.info(f'Clean overall accuracy: {overall:.4f}')

    preds_adv: List[torch.Tensor] = []
    preds_def: Dict[float, List[torch.Tensor]] = {float(p): [] for p in percents}
    for sample, label in zip(torch.chunk(sig_sel, cfg.test_batch_size, dim=0),
                             torch.chunk(lab_sel, cfg.test_batch_size, dim=0)):
        sample = sample.to(cfg.device)
        label = label.to(cfg.device)
        if getattr(cfg, 'attack', 'cw') == 'cw':
            adv = _cw_attack_batch(model, sample, label, cfg, logger)
        else:
            adv = _spectral_attack_batch(sample, cfg)
        with torch.no_grad():
            logits_adv, _ = model(adv)
            preds_adv.append(torch.argmax(logits_adv, dim=1).cpu())
            for pct in percents:
                adv_def = fft_topk_percent_denoise(adv, percent=float(pct))
                logits_def, _ = model(adv_def)
                preds_def[float(pct)].append(torch.argmax(logits_def, dim=1).cpu())

    preds_adv_cat = torch.cat(preds_adv, dim=0).numpy()
    overall_adv, per_mod_adv = _acc_by_mod(preds_adv_cat, lab_sel.numpy(), cfg.classes)
    results['cw'] = {'overall': overall_adv, **per_mod_adv}
    logger.info(f'CW attack accuracy: {overall_adv:.4f}')

    for pct, pred_list in preds_def.items():
        preds_pct = torch.cat(pred_list, dim=0).numpy()
        overall_pct, per_mod_pct = _acc_by_mod(preds_pct, lab_sel.numpy(), cfg.classes)
        name = f'cw_top{int(float(pct) * 100)}'
        results[name] = {'overall': overall_pct, **per_mod_pct}
        per_mod_str = ', '.join([f'{k}={v:.4f}' for k, v in sorted(per_mod_pct.items())])
        logger.info(f'{name} accuracy: overall={overall_pct:.4f} | {per_mod_str}')

    out_dir = os.path.join(cfg.result_dir, 'freq_topk_adv')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'freq_topk_adv_eval.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Saved results to {out_path}')

    return results
