import argparse
import os.path
import sys

import numpy as np
import torch

from data_loader.data_loader import Create_Data_Loader, Load_Dataset, Dataset_Split
from util.config import Config, merge_args2cfg
from util.evaluation import Run_Eval
from util.adv_eval import Run_Adv_Eval
from util.training import Trainer
from util.utils import fix_seed, log_exp_settings, create_AWN_model, create_model
from util.logger import create_logger
# Lazy import visualize utilities only when needed to avoid optional deps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')  # train ,eval or visualize
    parser.add_argument('--model', type=str, default='awn', help='Model architecture: awn or mcldnn')
    parser.add_argument('--dataset', type=str, default='2016.10a')  # 2016.10a, 2016.10b, 2018.01a
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--Draw_Confmat', type=bool, default=True)
    parser.add_argument('--Draw_Acc_Curve', type=bool, default=True)
    # Optional filtering to speed up eval
    parser.add_argument('--mod_filter', type=str, default=None, help='e.g., QAM16')
    parser.add_argument('--snr_filter', type=int, default=None, help='e.g., 18')
    # Adversarial eval options
    parser.add_argument('--attack', type=str, default='cw')
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_kappa', type=float, default=0.0)
    parser.add_argument('--cw_steps', type=int, default=100)
    parser.add_argument('--cw_lr', type=float, default=1e-2)
    parser.add_argument('--cw_targeted', type=bool, default=False)
    parser.add_argument('--cw_scale', type=float, default=None, help='Optional post-scale 0..1 to reduce CW delta magnitude')
    parser.add_argument('--lowpass', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply lowpass smoothing to CW perturbation')
    parser.add_argument('--lowpass_kernel', type=int, default=17)
    # Spectral noise (no optimizer) options
    parser.add_argument('--spec_type', type=str, default='cw_tone', help='spectral profile: cw_tone | psd_band | psd_mask')
    parser.add_argument('--spec_eps', type=float, default=0.1, help='L2 norm of spectral perturbation')
    parser.add_argument('--spec_jnr_db', type=float, default=None, help='Alternative to spec_eps: JNR in dB')
    parser.add_argument('--tone_freq', type=float, default=None, help='Normalized tone freq in [0,0.5]')
    parser.add_argument('--spec_band_low', type=float, default=0.05, help='Lower edge for psd_band in [0,0.5]')
    parser.add_argument('--spec_band_high', type=float, default=0.25, help='Upper edge for psd_band in [0,0.5]')
    parser.add_argument('--spec_mask_path', type=str, default=None, help='Path to .npy one-sided PSD mask (len T//2+1)')
    parser.add_argument('--eval_limit', type=int, default=None, help='Limit per-SNR test examples for (adv_)eval')
    parser.add_argument('--attack_backend', type=str, default='torchattacks', help='torchattacks (default) or internal')
    parser.add_argument('--ta_box', type=str, default='unit', help='torchattacks input mapping: unit|minmax')
    parser.add_argument('--spec_mask_out', type=str, default=None, help='Output path for build_psd_mask mode')
    # Optional defense (FFT-domain recovery)
    parser.add_argument('--def_band_low', type=float, default=None, help='Lower edge for fft_notch in [0,0.5]')
    parser.add_argument('--def_band_high', type=float, default=None, help='Upper edge for fft_notch in [0,0.5]')
    parser.add_argument('--def_mask_path', type=str, default=None, help='Path to .npy keep-mask (len T//2+1) for fft_mask')
    parser.add_argument('--cmp_defense', type=bool, default=False, help='Compare accuracy before/after defense')
    parser.add_argument('--def_notch_depth', type=float, default=1.0, help='Soft notch depth (0..1), 1=full suppression')
    parser.add_argument('--def_notch_trans', type=int, default=3, help='Transition width (bins) for soft notch')
    parser.add_argument('--def_hp_order', type=int, default=1, help='Order for highpass_diff (1 or 2)')
    # Adaptive soft-notch / detrend params
    parser.add_argument('--def_auto_fmax', type=float, default=0.08, help='Max freq for auto notch search')
    parser.add_argument('--def_auto_ref_low', type=float, default=0.15, help='Ref band low for baseline')
    parser.add_argument('--def_auto_ref_high', type=float, default=0.5, help='Ref band high for baseline')
    parser.add_argument('--def_auto_tau', type=float, default=2.0, help='Threshold ratio peak/baseline to notch')
    parser.add_argument('--def_auto_max_width', type=int, default=3, help='Max half-width (bins) for auto notch')
    parser.add_argument('--def_auto_depth_max', type=float, default=0.8, help='Max depth for auto notch (0..1)')
    parser.add_argument('--def_auto_trans', type=int, default=4, help='Transition bins for auto notch')
    # Ensemble soft notch
    parser.add_argument('--defense', type=str, default='none',
                        help='Defense to apply before inference: none | fft_notch | fft_mask | idfft_notch | fft_soft_notch | highpass_diff | auto_soft_notch | dc_detrend | fft_soft_notch_ens | fft_topk | fft_topk_percent | ae_fft_topk')
    parser.add_argument('--def_ens_depths', type=str, default='0.55,0.6,0.65',
                        help='Comma-separated depths for fft_soft_notch_ens (e.g., "0.55,0.6,0.65")')
    parser.add_argument('--def_ens_trans', type=int, default=4, help='Transition bins for fft_soft_notch_ens')
    # Top-K FFT and AE-detector options (AWN_All-style)
    parser.add_argument('--def_topk', type=int, default=50, help='K for fft_topk/ae_fft_topk denoiser')
    parser.add_argument('--def_topk_percent', type=float, default=None, help='Percent (0..1] for fft_topk_percent')
    parser.add_argument('--detector_ckpt', type=str, default=None, help='Path to AE detector .pth/.pt (optional)')
    parser.add_argument('--detector_threshold', type=float, default=4.468164592981338e-03, help='KL threshold for AE gate')
    parser.add_argument('--detector_norm_offset', type=float, default=0.02, help='Detector normalization offset')
    parser.add_argument('--detector_norm_scale', type=float, default=0.04, help='Detector normalization scale')
    # Detector training/calibration
    parser.add_argument('--det_epochs', type=int, default=10)
    parser.add_argument('--det_batch_size', type=int, default=256)
    parser.add_argument('--det_lr', type=float, default=1e-3)
    parser.add_argument('--det_wd', type=float, default=0.0)
    parser.add_argument('--det_patience', type=int, default=5)
    parser.add_argument('--det_train_limit', type=int, default=20000, help='Limit number of clean training samples for AE')
    parser.add_argument('--det_calib_quantile', type=float, default=0.90, help='KL quantile for threshold calibration')
    # Frequency top-k eval
    parser.add_argument('--snr_min', type=float, default=None, help='Minimum SNR to keep (e.g., 0 for >=0)')
    parser.add_argument('--freq_percents', type=str, default='0.1,0.2,0.3,0.4,0.5',
                        help='Comma-separated percents (0..1] for fft_topk_percent in freq_topk_eval mode')
    parser.add_argument('--dir_name', type=str, default=None, help='Directory name for experiment')
    # Multi-attack evaluation
    parser.add_argument('--attack_list', type=str, default=None,
                        help='Comma-separated attack names for multi_attack_eval (default: all 15)')
    parser.add_argument('--eval_limit_per_cell', type=int, default=None,
                        help='Max samples per (SNR, mod) cell for multi_attack_eval')
    parser.add_argument('--attack_eps', type=float, default=0.03,
                        help='Epsilon for Linf attacks (default: 0.03 for IQ data)')
    parser.add_argument('--plot_freq', action='store_true',
                        help='Plot frequency domain comparison (clean vs adversarial)')
    parser.add_argument('--plot_iq', action='store_true',
                        help='Plot IQ distribution comparison (clean vs adversarial)')
    parser.add_argument('--plot_n_samples', type=int, default=3,
                        help='Number of individual samples to plot for freq/IQ comparison')
    # SigGuard evaluation params
    parser.add_argument('--sigguard_topk', type=int, default=50, help='Top-K for SigGuard defense')
    parser.add_argument('--no_plot_iq', action='store_true', help='Disable IQ distribution plots in sigguard_eval')
    # EAD attack params (EADL1, EADEN)
    parser.add_argument('--ead_kappa', type=float, default=0, help='EAD confidence/kappa parameter')
    parser.add_argument('--ead_lr', type=float, default=0.01, help='EAD learning rate')
    parser.add_argument('--ead_max_iterations', type=int, default=100, help='EAD max iterations')
    parser.add_argument('--ead_binary_search_steps', type=int, default=9, help='EAD binary search steps')
    parser.add_argument('--ead_initial_const', type=float, default=0.001, help='EAD initial constant')
    parser.add_argument('--ead_beta', type=float, default=0.001, help='EAD beta (L1/L2 tradeoff)')
    args = parser.parse_args()

    fix_seed(args.seed)

    # Parse freq percents early so cfg has a list
    freq_pct = getattr(args, 'freq_percents', None)
    if isinstance(freq_pct, str):
        args.freq_percents = [float(p) for p in freq_pct.split(',') if p]

    # Default ta_box to minmax for MCLDNN (LSTM models need stronger perturbations)
    model_name = args.model.lower()
    if model_name == 'mcldnn' and '--ta_box' not in sys.argv:
        args.ta_box = 'minmax'

    cfg = Config(args.dataset, train=(args.mode == 'train'))
    cfg.init_dir(args.dir_name)
    cfg = merge_args2cfg(cfg, vars(args))
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    log_exp_settings(logger, cfg)

    model = create_model(cfg, model_name)
    logger.info(f">>> Model: {model_name.upper()}")
    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    # Helper to get checkpoint filename
    def get_ckpt_name():
        return f"{cfg.dataset}_{model_name.upper()}.pkl"

    Signals, Labels, SNRs, snrs, mods = Load_Dataset(cfg.dataset, logger, mod_filter=args.mod_filter, snr_filter=args.snr_filter)
    train_set, test_set, val_set, test_idx = Dataset_Split(
        Signals,
        Labels,
        snrs,
        mods,
        logger)
    Signals_test, Labels_test = test_set

    if args.mode == 'train':
        train_loader, val_loader = Create_Data_Loader(train_set, val_set, cfg, logger)
        trainer = Trainer(model,
                          train_loader,
                          val_loader,
                          cfg,
                          logger,
                          model_name=model_name)
        trainer.loop()
        from util.visualize import save_training_process
        save_training_process(trainer.epochs_stats, cfg)

        save_model_name = get_ckpt_name()
        model.load_state_dict(torch.load(os.path.join(cfg.model_dir, save_model_name), map_location=cfg.device))
        Run_Eval(model,
                 Signals_test,
                 Labels_test,
                 SNRs,
                 test_idx,
                 cfg,
                 logger)

    elif args.mode == 'eval':
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, get_ckpt_name()), map_location=cfg.device))
        Run_Eval(model,
                 Signals_test,
                 Labels_test,
                 SNRs,
                 test_idx,
                 cfg,
                 logger)

    elif args.mode == 'visualize':
        from util.visualize import Visualize_LiftingScheme
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, get_ckpt_name()), map_location=cfg.device))
        for i in range(0, 8):
            index = np.random.randint(0, Signals_test.shape[0])
            test_sample = Signals_test[index]
            test_sample = test_sample[np.newaxis, ...]
            Visualize_LiftingScheme(model, test_sample, cfg, index)

    elif args.mode == 'adv_eval':
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, get_ckpt_name()), map_location=cfg.device))
        Run_Adv_Eval(model,
                     Signals_test,
                     Labels_test,
                     SNRs,
                     test_idx,
                     cfg,
                     logger)

    elif args.mode == 'freq_compare':
        from util.freq_compare import run_freq_compare
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, get_ckpt_name()), map_location=cfg.device))
        run_freq_compare(
            model,
            Signals_test,
            Labels_test,
            SNRs,
            test_idx,
            cfg,
            logger,
            tone_freq=args.tone_freq,
            spec_eps=args.spec_eps,
            spec_type=args.spec_type,
            spec_mask_path=args.spec_mask_path,
            spec_band_low=args.spec_band_low,
            spec_band_high=args.spec_band_high,
        )
    elif args.mode == 'freq_topk_eval':
        from util.freq_topk_eval import run_freq_topk_eval
        model_path = os.path.join(args.ckpt_path, get_ckpt_name())
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        run_freq_topk_eval(
            model,
            Signals_test,
            Labels_test,
            SNRs,
            test_idx,
            cfg,
            logger,
            snr_min=(
                getattr(args, 'snr_min', 0.0)
                if getattr(args, 'snr_min', None) is not None
                else 0.0
            ),
            percents=getattr(args, 'freq_percents', (0.1, 0.2, 0.3, 0.4, 0.5)),
            eval_limit=getattr(args, 'eval_limit', None),
        )
    elif args.mode == 'freq_topk_adv_eval':
        from util.freq_topk_adv_eval import run_freq_topk_adv_eval
        model_path = os.path.join(args.ckpt_path, get_ckpt_name())
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        run_freq_topk_adv_eval(
            model,
            Signals_test,
            Labels_test,
            SNRs,
            test_idx,
            cfg,
            logger,
            snr_min=(
                getattr(args, 'snr_min', 0.0)
                if getattr(args, 'snr_min', None) is not None
                else 0.0
            ),
            percents=getattr(args, 'freq_percents', (0.1, 0.2, 0.3, 0.4, 0.5)),
            eval_limit=getattr(args, 'eval_limit', None),
        )
    elif args.mode == 'train_detector':
        # Train AE detector on clean training/val splits (normalized MSE)
        from util.detector_train import train_detector
        save_path = os.path.join(args.ckpt_path, 'detector_ae.pth')
        # Use limits for speed if requested
        limit = getattr(args, 'det_train_limit', None)
        # Use split tensors directly
        x_train, _ = train_set
        x_val, _ = val_set
        if limit is not None:
            x_train = x_train[:limit]
        ckpt = train_detector(
            x_train,
            x_val,
            device=cfg.device,
            out_path=save_path,
            epochs=getattr(args, 'det_epochs', 10),
            batch_size=getattr(args, 'det_batch_size', 256),
            lr=getattr(args, 'det_lr', 1e-3),
            weight_decay=getattr(args, 'det_wd', 0.0),
            patience=getattr(args, 'det_patience', 5),
            num_workers=getattr(args, 'num_workers', 0),
            logger=logger,
        )
        logger.info(f"Saved AE detector to: {ckpt}")
    elif args.mode == 'calibrate_detector':
        # Compute KL on clean validation set and suggest a threshold
        from util.detector_train import calibrate_threshold
        x_val, _ = val_set
        if args.detector_ckpt is None:
            raise SystemExit('Please provide --detector_ckpt for calibration')
        thr = calibrate_threshold(
            x_val,
            args.detector_ckpt,
            device=cfg.device,
            quantile=getattr(args, 'det_calib_quantile', 0.90),
            batch_size=getattr(args, 'det_batch_size', 256),
            num_workers=getattr(args, 'num_workers', 0),
            logger=logger,
        )
        print(f'Recommended detector_threshold: {thr:.6f}')

    elif args.mode == 'build_psd_mask':
        # Build an average PSD mask from the filtered dataset split (test set)
        from util.psd_tools import compute_avg_psd_mask
        import numpy as np
        os.makedirs(cfg.result_dir, exist_ok=True)
        limit = args.eval_limit
        x = Signals_test[:limit] if limit is not None else Signals_test
        mask = compute_avg_psd_mask(x)
        out_path = getattr(args, 'spec_mask_out', None)
        if out_path is None:
            # Default path uses filters in name if present
            mf = args.mod_filter or 'ALL'
            sf = args.snr_filter if args.snr_filter is not None else 'ALL'
            out_path = os.path.join(cfg.result_dir, f'psd_mask_{mf}_{sf}.npy')
        np.save(out_path, mask)
        print(f'Saved PSD mask to: {out_path} (len={len(mask)})')

    elif args.mode == 'build_cw_psd_mask':
        # Build an average PSD mask specifically from CW perturbations (adv - x)
        from util.psd_tools import compute_avg_psd_mask
        from util.adv_attack import cw_l2_attack
        import numpy as np
        os.makedirs(cfg.result_dir, exist_ok=True)
        limit = args.eval_limit
        x = Signals_test[:limit] if limit is not None else Signals_test
        y = Labels_test[:limit] if limit is not None else Labels_test
        # Chunk for memory
        Sample = torch.chunk(x, cfg.test_batch_size, dim=0)
        Label = torch.chunk(y, cfg.test_batch_size, dim=0)
        deltas = []
        for (sample, label) in zip(Sample, Label):
            sample = sample.to(cfg.device)
            label = label.to(cfg.device)
            adv = cw_l2_attack(
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
            deltas.append((adv - sample).cpu())
        delta = torch.cat(deltas, dim=0)
        mask = compute_avg_psd_mask(delta)
        out_path = getattr(args, 'spec_mask_out', None)
        if out_path is None:
            mf = args.mod_filter or 'ALL'
            sf = args.snr_filter if args.snr_filter is not None else 'ALL'
            out_path = os.path.join(cfg.result_dir, f'cw_psd_mask_{mf}_{sf}.npy')
        np.save(out_path, mask)
        print(f'Saved CW-perturbation PSD mask to: {out_path} (len={len(mask)})')

    elif args.mode == 'adv_bench':
        from util.bench import run_attack_bench
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, get_ckpt_name()), map_location=cfg.device))
        run_attack_bench(
            model,
            Signals_test,
            Labels_test,
            cfg,
            logger,
        )

    elif args.mode == 'multi_attack_eval':
        from util.multi_attack_eval import run_multi_attack_snr_mod_eval
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, get_ckpt_name()), map_location=cfg.device))
        # Parse attack list if provided
        attack_list = None
        if args.attack_list is not None:
            attack_list = [a.strip() for a in args.attack_list.split(',') if a.strip()]
        run_multi_attack_snr_mod_eval(
            model,
            Signals_test,
            Labels_test,
            SNRs,
            test_idx,
            cfg,
            logger,
            attacks=attack_list,
            eval_limit_per_cell=args.eval_limit_per_cell,
            plot_freq=args.plot_freq,
            plot_iq=args.plot_iq,
            plot_n_samples=args.plot_n_samples,
        )

    elif args.mode == 'sigguard_eval':
        from util.sigguard_eval import run_sigguard_eval
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, get_ckpt_name()), map_location=cfg.device))
        # Parse attack list if provided
        attack_list = None
        if args.attack_list is not None:
            attack_list = [a.strip() for a in args.attack_list.split(',') if a.strip()]
        # IQ plots enabled by default for sigguard_eval (use --no_plot_iq to disable)
        should_plot_iq = not getattr(args, 'no_plot_iq', False)
        run_sigguard_eval(
            model,
            Signals_test,
            Labels_test,
            cfg,
            logger,
            attacks=attack_list,
            topk=args.sigguard_topk,
            eval_limit=args.eval_limit,
            plot_iq=should_plot_iq,
            plot_n_samples=args.plot_n_samples,
        )
