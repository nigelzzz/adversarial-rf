#!/usr/bin/env python
"""
Test CW attack with recovery on SNR >= 0 samples.

This script tests:
1. CW attack using torchattacks on high-SNR samples (SNR >= 0)
2. FFT Top-K recovery (AWN_All.py pattern)
3. Optional detector-gated recovery

Usage:
    python test_cw_snr0.py --dataset 2016.10a
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm

from util.utils import create_AWN_model, fix_seed
from util.config import Config, merge_args2cfg
from data_loader.data_loader import Load_Dataset, Dataset_Split
from util.defense import fft_topk_denoise, normalize_iq_data, denormalize_iq_data
from util.detector import RFSignalAutoEncoder, detector_gate_fft_topk


def filter_by_snr(signals, labels, snrs, snr_threshold=0):
    """Filter samples to keep only those with SNR >= threshold."""
    mask = snrs >= snr_threshold
    return signals[mask], labels[mask], snrs[mask]


def main():
    parser = argparse.ArgumentParser(description='Test CW attack with recovery on SNR >= 0')
    parser.add_argument('--dataset', type=str, default='2016.10a')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to AWN classifier checkpoint (if None, looks in ./)')
    parser.add_argument('--detector_ckpt', type=str, default=None,
                        help='Path to detector checkpoint (optional)')
    parser.add_argument('--defense', type=str, default='fft_topk',
                        choices=['fft_topk', 'ae_fft_topk', 'none'])
    parser.add_argument('--def_topk', type=int, default=50)
    parser.add_argument('--detector_threshold', type=float, default=0.004468)
    parser.add_argument('--cw_steps', type=int, default=100)
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--snr_threshold', type=int, default=0, help='SNR threshold')
    parser.add_argument('--snr_strict_gt', action='store_true', help='Use strict SNR > threshold (default >=)')
    parser.add_argument('--mods', type=str, default='BPSK,QPSK,QAM16,QAM64',
                        help='Comma-separated modulation names to include')
    parser.add_argument('--test_samples', type=int, default=None,
                        help='Max test samples (None = all SNR>=0 samples)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device)

    # Load config and model
    cfg = Config(args.dataset, train=False)
    cfg = merge_args2cfg(cfg, vars(args))

    model = create_AWN_model(cfg)

    # Try to find checkpoint
    if args.ckpt_path is None:
        import glob
        pattern = f'./{args.dataset}_AWN.pkl'
        matches = glob.glob(pattern)
        if matches:
            args.ckpt_path = matches[0]
        else:
            pattern2 = f'./*{args.dataset}*.pkl'
            matches2 = glob.glob(pattern2)
            if matches2:
                args.ckpt_path = matches2[0]
            else:
                print(f"ERROR: No checkpoint found. Please specify --ckpt_path")
                return

    try:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        print(f"✓ Loaded AWN classifier from {args.ckpt_path}")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return

    # Load detector if using ae_fft_topk
    detector = None
    if args.defense == 'ae_fft_topk':
        if args.detector_ckpt:
            try:
                detector = RFSignalAutoEncoder().to(device)
                det_ckpt = torch.load(args.detector_ckpt, map_location=device)
                detector.load_state_dict(det_ckpt)
                detector.eval()
                print(f"✓ Loaded detector from {args.detector_ckpt}")
            except Exception as e:
                print(f"WARNING: Could not load detector ({e}), using fft_topk instead")
                args.defense = 'fft_topk'
        else:
            print("WARNING: ae_fft_topk requires --detector_ckpt, using fft_topk instead")
            args.defense = 'fft_topk'

    # Load test data
    # Load dataset directly from pickle (robust to logger issues)
    print(f"\nLoading {args.dataset} dataset...")
    import pickle as _pkl
    ds_path = './data/RML2016.10a_dict.pkl' if args.dataset == '2016.10a' else None
    if ds_path is None:
        raise SystemExit('test_cw_snr0.py currently supports dataset=2016.10a')
    with open(ds_path, 'rb') as f:
        Xd = _pkl.load(f, encoding='bytes')
    snrs_list, mods_list_all = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    # Select mods
    mods_sel = [m.strip().encode() for m in args.mods.split(',') if m.strip()]
    mods_list = [m for m in mods_list_all if m in mods_sel]

    # Stack all into arrays
    import numpy as _np
    X, lbl, snr_vals = [], [], []
    for m in mods_list:
        for s in snrs_list:
            arr = Xd[(m, s)]
            X.append(arr)
            # Use AWN's label mapping to match training (avoid mismatches)
            lbl.extend([cfg.classes[m]] * arr.shape[0])
            snr_vals.extend([s] * arr.shape[0])
    X = _np.vstack(X)
    lbl = _np.array(lbl)
    snr_vals = _np.array(snr_vals)

    # Filter by SNR >= or > threshold
    if args.snr_strict_gt:
        mask = snr_vals > args.snr_threshold
    else:
        mask = snr_vals >= args.snr_threshold
    sig_test_np = X[mask]
    lab_test_np = lbl[mask]
    snr_test = snr_vals[mask]
    op = '>' if args.snr_strict_gt else '>='
    print(f"✓ Filtered {len(X)} samples → {len(sig_test_np)} with SNR {op} {args.snr_threshold}")
    print(f"  SNR range: {snr_test.min():.0f} to {snr_test.max():.0f} dB")

    # Stratified sampling across all SNR >= threshold
    if args.test_samples is not None and len(sig_test_np) > args.test_samples:
        unique_snrs = sorted(_np.unique(snr_test))
        per = max(1, args.test_samples // len(unique_snrs))
        keep_idx = []
        for s in unique_snrs:
            idx = _np.where(snr_test == s)[0]
            if len(idx) == 0:
                continue
            sel = _np.random.choice(idx, size=min(per, len(idx)), replace=False)
            keep_idx.append(sel)
        keep_idx = _np.concatenate(keep_idx)
        # If we are short, top up randomly from remaining
        if keep_idx.size < args.test_samples:
            remaining = _np.setdiff1d(_np.arange(len(sig_test_np)), keep_idx)
            extra = _np.random.choice(remaining, size=min(args.test_samples - keep_idx.size, remaining.size), replace=False)
            keep_idx = _np.concatenate([keep_idx, extra])
        # Shuffle for randomness
        rng = _np.random.default_rng(42)
        rng.shuffle(keep_idx)
        sig_test_np = sig_test_np[keep_idx]
        lab_test_np = lab_test_np[keep_idx]
        snr_test = snr_test[keep_idx]
        print(f"  Stratified to {len(sig_test_np)} samples across SNRs: {unique_snrs}")

    # Torch tensors
    sig_test = torch.from_numpy(sig_test_np).float()
    lab_test = torch.from_numpy(lab_test_np).long()

    # Import torchattacks
    try:
        import torchattacks
        print(f"✓ Using torchattacks {torchattacks.__version__} for CW attack")
    except ImportError:
        print("ERROR: torchattacks not installed. Run: pip install torchattacks")
        return

    # Run CW attack
    print(f"\n{'='*70}")
    print(f"Running CW attack (steps={args.cw_steps}, c={args.cw_c})...")
    print(f"{'='*70}")

    from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq
    wrapped = Model01Wrapper(model).eval()
    atk = torchattacks.CW(wrapped, c=args.cw_c, kappa=0.0, steps=args.cw_steps, lr=1e-2)

    batch_size = 50
    adv_all = []
    clean_preds = []
    adv_preds = []
    recovered_preds = []
    labels_all = []

    num_batches = (len(sig_test) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(sig_test), batch_size), total=num_batches, desc="Processing batches"):
        batch_sig = sig_test[i:i+batch_size].to(device)
        batch_lab = lab_test[i:i+batch_size].to(device)

        # Clean predictions
        with torch.no_grad():
            logits_clean, _ = model(batch_sig)
            pred_clean = torch.argmax(logits_clean, dim=1)
            clean_preds.append(pred_clean.cpu())

        # Generate adversarial examples with 01/4D mapping
        batch_x01_4d = iq_to_ta_input(batch_sig)
        adv01_4d = atk(batch_x01_4d, batch_lab)
        adv = ta_output_to_iq(adv01_4d)
        adv_all.append(adv.cpu())

        # Adversarial predictions (before recovery)
        with torch.no_grad():
            logits_adv, _ = model(adv)
            pred_adv = torch.argmax(logits_adv, dim=1)
            adv_preds.append(pred_adv.cpu())

        # Apply recovery
        if args.defense == 'ae_fft_topk' and detector is not None:
            with torch.no_grad():
                adv_recovered, kl_vals = detector_gate_fft_topk(
                    adv, detector,
                    threshold=args.detector_threshold,
                    topk=args.def_topk,
                    norm_offset=0.02,
                    norm_scale=0.04,
                    apply_in_normalized=True
                )
        elif args.defense == 'fft_topk':
            # Simple FFT Top-K without detector (AWN_All.py pattern)
            with torch.no_grad():
                adv_norm = normalize_iq_data(adv, 0.02, 0.04)
                adv_filt = fft_topk_denoise(adv_norm, topk=args.def_topk)
                adv_recovered = denormalize_iq_data(adv_filt, 0.02, 0.04)
        else:
            # No defense
            adv_recovered = adv

        # Recovered predictions
        with torch.no_grad():
            logits_rec, _ = model(adv_recovered)
            pred_rec = torch.argmax(logits_rec, dim=1)
            recovered_preds.append(pred_rec.cpu())

        labels_all.append(batch_lab.cpu())

    # Compute accuracies
    clean_preds = torch.cat(clean_preds)
    adv_preds = torch.cat(adv_preds)
    recovered_preds = torch.cat(recovered_preds)
    labels = torch.cat(labels_all)

    clean_acc = (clean_preds == labels).float().mean().item() * 100
    adv_acc = (adv_preds == labels).float().mean().item() * 100
    recovered_acc = (recovered_preds == labels).float().mean().item() * 100

    print(f"\n{'='*70}")
    print(f"RESULTS on {len(labels)} samples (SNR >= {args.snr_threshold} dB)")
    print(f"{'='*70}")
    print(f"  Clean Accuracy:        {clean_acc:6.2f}%")
    print(f"  Adversarial Accuracy:  {adv_acc:6.2f}%  (attack success: {clean_acc - adv_acc:+.2f}%)")
    print(f"  Recovered Accuracy:    {recovered_acc:6.2f}%  (defense: {args.defense}, topk={args.def_topk})")
    print(f"  Recovery Gain:         {recovered_acc - adv_acc:+6.2f}%")
    print(f"{'='*70}")

    # Per-SNR breakdown
    unique_snrs = sorted(np.unique(snr_test))
    print(f"\nPer-SNR Breakdown:")
    print(f"{'SNR (dB)':<10} {'Clean %':<10} {'Adv %':<10} {'Recovered %':<12} {'Recovery Gain'}")
    print(f"{'-'*70}")

    for snr_val in unique_snrs:
        mask = snr_test == snr_val
        mask_t = torch.from_numpy(mask)

        if mask.sum() == 0:
            continue

        clean_acc_snr = (clean_preds[mask_t] == labels[mask_t]).float().mean().item() * 100
        adv_acc_snr = (adv_preds[mask_t] == labels[mask_t]).float().mean().item() * 100
        rec_acc_snr = (recovered_preds[mask_t] == labels[mask_t]).float().mean().item() * 100
        gain = rec_acc_snr - adv_acc_snr

        print(f"{snr_val:>6.0f}     {clean_acc_snr:>6.2f}%    {adv_acc_snr:>6.2f}%    {rec_acc_snr:>6.2f}%       {gain:+6.2f}%")

    # Save results
    results = {
        'clean_acc': clean_acc,
        'adv_acc': adv_acc,
        'recovered_acc': recovered_acc,
        'defense': args.defense,
        'topk': args.def_topk,
        'cw_steps': args.cw_steps,
        'cw_c': args.cw_c,
        'snr_threshold': args.snr_threshold,
        'num_samples': len(labels),
        'dataset': args.dataset,
    }

    output_path = f"cw_recovery_snr{args.snr_threshold}_{args.dataset}.pt"
    torch.save(results, output_path)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
