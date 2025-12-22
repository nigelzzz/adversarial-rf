import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score

from util.adv_attack import (
    cw_l2_attack,
    _lowpass_filter,
    _batch_clip,
    spectral_noise_attack,
    Model01Wrapper,
    iq_to_ta_input,
    ta_output_to_iq,
)
from util.defense import (
    fft_notch_denoise,
    fft_mask_denoise,
    fft_soft_notch_denoise,
    highpass_diff,
    auto_soft_notch_denoise,
    dc_detrend,
    fft_topk_denoise,
    fft_topk_percent_denoise,
    normalize_iq_data,
    denormalize_iq_data,
)
from typing import Optional


class _AWNModelWrapper(torch.nn.Module):
    """Wrapper to make AWN model compatible with torchattacks (returns only logits)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x)
        return logits


def Run_Adv_Eval(model,
                 sig_test,
                 lab_test,
                 SNRs,
                 test_idx,
                 cfg,
                 logger):

    model.eval()

    snrs = list(np.unique(SNRs))
    mods = list(cfg.classes.keys())

    Confmat_Set = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
    Accuracy_list = np.zeros(len(snrs), dtype=float)
    # Optional defense comparison containers
    use_defense = getattr(cfg, 'defense', 'none') is not None and getattr(cfg, 'defense', 'none') != 'none'
    compare_def = bool(getattr(cfg, 'cmp_defense', False)) and use_defense
    if compare_def:
        Confmat_Set_def = np.zeros((len(snrs), len(mods), len(mods)), dtype=int)
        Accuracy_list_def = np.zeros(len(snrs), dtype=float)
        pre_lab_all_def = []

    pre_lab_all = []
    label_all = []

    for snr_i, snr in enumerate(snrs):
        test_SNRs = list(map(lambda x: SNRs[x], test_idx))
        test_SNRs = np.array(test_SNRs).squeeze()
        test_sig_i = sig_test[np.where(np.array(test_SNRs) == snr)]
        test_lab_i = lab_test[np.where(np.array(test_SNRs) == snr)]

        # Optional limit for faster attack experimentation
        limit = getattr(cfg, 'eval_limit', None)
        if limit is not None:
            test_sig_i = test_sig_i[:limit]
            test_lab_i = test_lab_i[:limit]

        # Split set into chunks to control memory (mimics Run_Eval)
        Sample = torch.chunk(test_sig_i, cfg.test_batch_size, dim=0)
        Label = torch.chunk(test_lab_i, cfg.test_batch_size, dim=0)

        pred_i = []
        label_i = []
        pred_i_def = [] if compare_def else None
        for (sample, label) in zip(Sample, Label):
            sample = sample.to(cfg.device)
            label = label.to(cfg.device)

            if cfg.attack == 'cw':
                backend = getattr(cfg, 'attack_backend', 'torchattacks')
                if backend == 'torchattacks':
                    try:
                        import torchattacks
                        # Use 01+4D wrapper per gpt.md to respect torchattacks' [0,1] box-constraint
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
                            from util.adv_attack import iq_to_ta_input_minmax, ta_output_to_iq_minmax
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
                            # clip to observed batch range
                            clip_min = sample.amin(dim=(1, 2), keepdim=True)
                            clip_max = sample.amax(dim=(1, 2), keepdim=True)
                            adv = _batch_clip(sample + delta, clip_min, clip_max)
                        # Optional post scaling to reduce attack magnitude
                        cw_scale = getattr(cfg, 'cw_scale', None)
                        if cw_scale is not None:
                            try:
                                s = float(cw_scale)
                                if s < 1.0:
                                    delta = adv - sample
                                    adv = _batch_clip(sample + s * delta,
                                                      sample.amin(dim=(1, 2), keepdim=True),
                                                      sample.amax(dim=(1, 2), keepdim=True))
                            except Exception:
                                pass
                    except Exception as e:
                        logger.info(f"Falling back to internal CW due to: {e}")
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
                else:
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
                    # Optional post scaling
                    cw_scale = getattr(cfg, 'cw_scale', None)
                    if cw_scale is not None:
                        try:
                            s = float(cw_scale)
                            if s < 1.0:
                                delta = adv - sample
                                adv = _batch_clip(sample + s * delta,
                                                  sample.amin(dim=(1, 2), keepdim=True),
                                                  sample.amax(dim=(1, 2), keepdim=True))
                        except Exception:
                            pass
            elif cfg.attack == 'spectral':
                # Add spectrally-shaped perturbations without model-based optimization
                spec_type = getattr(cfg, 'spec_type', 'cw_tone')
                kwargs = dict(
                    spec_type=spec_type,
                    spec_eps=getattr(cfg, 'spec_eps', 0.1),
                    jnr_db=getattr(cfg, 'spec_jnr_db', None),
                    tone_freq=getattr(cfg, 'tone_freq', None),
                )
                # Optional band and mask
                if spec_type in ('psd_band', 'band'):
                    kwargs['band'] = (
                        getattr(cfg, 'spec_band_low', 0.05),
                        getattr(cfg, 'spec_band_high', 0.25),
                    )
                if spec_type in ('psd_mask', 'mask'):
                    mask_path = getattr(cfg, 'spec_mask_path', None)
                    if mask_path is not None:
                        import numpy as _np
                        try:
                            mask_np = _np.load(mask_path)
                            import torch as _torch
                            mask_t = _torch.as_tensor(mask_np)
                        except Exception as e:
                            logger.info(f"Failed to load PSD mask from {mask_path}: {e}")
                            mask_t = None
                    else:
                        mask_t = None
                    kwargs['psd_mask'] = mask_t

                adv = spectral_noise_attack(sample, **kwargs)
            else:
                raise NotImplementedError(f"Unknown attack: {cfg.attack}")

            # Optional FFT-domain defense
            adv_def = None
            if use_defense:
                dmode = str(getattr(cfg, 'defense', 'none')).lower()
                if dmode in ('fft_notch', 'idfft_notch'):
                    f_low = getattr(cfg, 'def_band_low', None)
                    f_high = getattr(cfg, 'def_band_high', None)
                    # Fall back to attack band if not provided
                    if f_low is None:
                        f_low = getattr(cfg, 'spec_band_low', 0.05)
                    if f_high is None:
                        f_high = getattr(cfg, 'spec_band_high', 0.25)
                    adv_def = fft_notch_denoise(adv, f_low, f_high)
                elif dmode in ('fft_soft_notch', 'soft_notch'):
                    f_low = getattr(cfg, 'def_band_low', None)
                    f_high = getattr(cfg, 'def_band_high', None)
                    if f_low is None:
                        f_low = getattr(cfg, 'spec_band_low', 0.05)
                    if f_high is None:
                        f_high = getattr(cfg, 'spec_band_high', 0.25)
                    depth = float(getattr(cfg, 'def_notch_depth', 1.0))
                    trans = int(getattr(cfg, 'def_notch_trans', 3))
                    adv_def = fft_soft_notch_denoise(adv, f_low, f_high, depth=depth, trans=trans)
                elif dmode == 'fft_mask':
                    mask_path = getattr(cfg, 'def_mask_path', None)
                    if mask_path is not None:
                        try:
                            import numpy as _np
                            import torch as _torch
                            mask_np = _np.load(mask_path)
                            mask_t = _torch.as_tensor(mask_np)
                        except Exception:
                            mask_t = None
                    else:
                        mask_t = None
                    if mask_t is not None:
                        adv_def = fft_mask_denoise(adv, mask_t)
                elif dmode in ('highpass_diff', 'hp_diff'):
                    order = int(getattr(cfg, 'def_hp_order', 1))
                    adv_def = highpass_diff(adv, order=order)
                elif dmode in ('auto_soft_notch', 'auto_notch'):
                    adv_def = auto_soft_notch_denoise(
                        adv,
                        fmax=float(getattr(cfg, 'def_auto_fmax', 0.08)),
                        ref_band=(
                            float(getattr(cfg, 'def_auto_ref_low', 0.15)),
                            float(getattr(cfg, 'def_auto_ref_high', 0.5)),
                        ),
                        tau=float(getattr(cfg, 'def_auto_tau', 2.0)),
                        max_width_bins=int(getattr(cfg, 'def_auto_max_width', 3)),
                        depth_max=float(getattr(cfg, 'def_auto_depth_max', 0.8)),
                        trans=int(getattr(cfg, 'def_auto_trans', 4)),
                    )
                elif dmode in ('dc_detrend', 'detrend'):
                    adv_def = dc_detrend(adv)
                elif dmode in ('fft_soft_notch_ens', 'soft_notch_ens'):
                    # Ensemble over multiple depths; average logits later
                    f_low = getattr(cfg, 'def_band_low', None)
                    f_high = getattr(cfg, 'def_band_high', None)
                    if f_low is None:
                        f_low = getattr(cfg, 'spec_band_low', 0.05)
                    if f_high is None:
                        f_high = getattr(cfg, 'spec_band_high', 0.25)
                    depths_str = str(getattr(cfg, 'def_ens_depths', '0.55,0.6,0.65'))
                    try:
                        depths = [float(d) for d in depths_str.split(',') if d.strip()]
                    except Exception:
                        depths = [0.55, 0.6, 0.65]
                    trans = int(getattr(cfg, 'def_ens_trans', 4))
                    adv_def_list = [
                        fft_soft_notch_denoise(adv, f_low, f_high, depth=depth, trans=trans)
                        for depth in depths
                    ]
                elif dmode in ('fft_topk', 'topk'):
                    # AWN_All-style: normalize → keep top-K FFT components → denormalize
                    topk = int(getattr(cfg, 'def_topk', 50))
                    norm_offset = float(getattr(cfg, 'detector_norm_offset', 0.02))
                    norm_scale = float(getattr(cfg, 'detector_norm_scale', 0.04))
                    adv_norm = normalize_iq_data(adv, norm_offset, norm_scale)
                    adv_filt = fft_topk_denoise(adv_norm, topk=topk)
                    adv_def = denormalize_iq_data(adv_filt, norm_offset, norm_scale)
                elif dmode in ('fft_topk_percent', 'topk_percent'):
                    pct = float(getattr(cfg, 'def_topk_percent', 0.1))
                    # guard
                    pct = min(max(pct, 1.0 / adv.size(-1)), 1.0)
                    adv_def = fft_topk_percent_denoise(adv, percent=pct)
                elif dmode in ('ae_fft_topk', 'detector_topk'):
                    # AE-based gate: denoise only if KL divergence exceeds a threshold
                    # Lazy import to avoid optional dependency when unused
                    from util.detector import RFSignalAutoEncoder, detector_gate_fft_topk
                    # Cache detector on cfg to avoid reloading
                    det: Optional[RFSignalAutoEncoder] = getattr(cfg, '_detector_model', None)
                    if det is None:
                        ckpt = getattr(cfg, 'detector_ckpt', None)
                        det = RFSignalAutoEncoder().to(cfg.device)
                        loaded = False
                        if ckpt is not None and len(str(ckpt)) > 0 and ckpt != 'None':
                            try:
                                det.load_state_dict(torch.load(ckpt, map_location=cfg.device))
                                loaded = True
                            except Exception as e:
                                logger.info(f"Failed to load detector checkpoint {ckpt}: {e}")
                        det.eval()
                        setattr(cfg, '_detector_model', det)
                        setattr(cfg, '_detector_loaded', loaded)
                    loaded = bool(getattr(cfg, '_detector_loaded', False))
                    topk = int(getattr(cfg, 'def_topk', 50))
                    if loaded:
                        thr = float(getattr(cfg, 'detector_threshold', 4.468164592981338e-03))
                        adv_def, kl_vals = detector_gate_fft_topk(
                            adv,
                            det,
                            threshold=thr,
                            topk=topk,
                            norm_offset=float(getattr(cfg, 'detector_norm_offset', 0.02)),
                            norm_scale=float(getattr(cfg, 'detector_norm_scale', 0.04)),
                            apply_in_normalized=True,
                        )
                        # Optionally log share of defended samples
                        try:
                            ratio = (kl_vals > thr).float().mean().item()
                            logger.info(f"[ae_fft_topk] defended ratio in batch: {ratio:.2f}")
                        except Exception:
                            pass
                    else:
                        logger.info("Detector checkpoint not provided; falling back to fft_topk on all samples.")
                        adv_def = fft_topk_denoise(adv, topk=topk)
                else:
                    adv_def = None

            with torch.no_grad():
                logits, _ = model(adv)
                pre_lab = torch.argmax(logits, dim=1).cpu()
                pred_i.append(pre_lab)
                label_i.append(label.cpu())
                if compare_def:
                    if dmode in ('fft_soft_notch_ens', 'soft_notch_ens'):
                        # Average logits across ensemble outputs
                        logits_sum = None
                        for adv_d in adv_def_list:
                            l, _ = model(adv_d)
                            logits_sum = l if logits_sum is None else (logits_sum + l)
                        logits_ens = logits_sum / len(adv_def_list)
                        pre_lab_def = torch.argmax(logits_ens, dim=1).cpu()
                        pred_i_def.append(pre_lab_def)
                    elif adv_def is not None:
                        logits_def, _ = model(adv_def)
                        pre_lab_def = torch.argmax(logits_def, dim=1).cpu()
                        pred_i_def.append(pre_lab_def)

        pred_i = np.concatenate(pred_i)
        label_i = np.concatenate(label_i)

        pre_lab_all.append(pred_i)
        label_all.append(label_i)

        Confmat_Set[snr_i, :, :] = confusion_matrix(label_i, pred_i, labels=list(range(len(cfg.classes))))
        Accuracy_list[snr_i] = accuracy_score(label_i, pred_i)

        if compare_def and pred_i_def is not None and len(pred_i_def) > 0:
            pred_i_def = np.concatenate(pred_i_def)
            pre_lab_all_def.append(pred_i_def)
            Confmat_Set_def[snr_i, :, :] = confusion_matrix(label_i, pred_i_def, labels=list(range(len(cfg.classes))))
            Accuracy_list_def[snr_i] = accuracy_score(label_i, pred_i_def)

    pre_lab_all = np.concatenate(pre_lab_all)
    label_all = np.concatenate(label_all)

    F1_score = f1_score(label_all, pre_lab_all, average='macro')
    kappa = cohen_kappa_score(label_all, pre_lab_all)
    acc = np.mean(Accuracy_list)

    logger.info(f'[Adversarial] overall accuracy is: {acc}')
    logger.info(f'[Adversarial] macro F1-score is: {F1_score}')
    logger.info(f'[Adversarial] kappa coefficient is: {kappa}')

    if compare_def:
        acc_def = np.mean(Accuracy_list_def)
        pre_lab_all_def_cat = np.concatenate(pre_lab_all_def) if len(pre_lab_all_def) > 0 else None
        F1_def = f1_score(label_all, pre_lab_all_def_cat, average='macro') if pre_lab_all_def_cat is not None else None
        kappa_def = cohen_kappa_score(label_all, pre_lab_all_def_cat) if pre_lab_all_def_cat is not None else None
        logger.info(f'[Defense {getattr(cfg, "defense", "")} ] overall accuracy is: {acc_def}')
        if F1_def is not None:
            logger.info(f'[Defense] macro F1-score is: {F1_def}')
        if kappa_def is not None:
            logger.info(f'[Defense] kappa coefficient is: {kappa_def}')

    # Reuse plotting toggles if desired
    if getattr(cfg, 'Draw_Confmat', False):
        try:
            from util.visualize import Draw_Confmat
            Draw_Confmat(Confmat_Set, snrs, cfg)
        except Exception as e:
            logger.info(f'Skip confmat plotting: {e}')
    if getattr(cfg, 'Draw_Acc_Curve', False):
        try:
            from util.visualize import Snr_Acc_Plot
            Snr_Acc_Plot(Accuracy_list, Confmat_Set, snrs, cfg)
        except Exception as e:
            logger.info(f'Skip acc plotting: {e}')
