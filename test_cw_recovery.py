#!/usr/bin/env python
"""
Test script for CW attack with recovery (AWN_All.py pattern).

This script demonstrates:
1. Loading a pretrained AWN classifier
2. Generating CW adversarial examples using torchattacks
3. Applying FFT Top-K recovery with optional detector gating
4. Comparing accuracy before and after recovery

Usage:
    python test_cw_recovery.py --dataset 2016.10a --ckpt_path ./2016.10a_AWN.pkl \
        --defense fft_topk --def_topk 50

    # With detector:
    python test_cw_recovery.py --dataset 2016.10a --ckpt_path ./2016.10a_AWN.pkl \
        --defense ae_fft_topk --def_topk 50 --detector_ckpt ./checkpoint/detector_ae.pth
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm

from util.utils import create_AWN_model
from util.config import Config, merge_args2cfg
from data_loader.data_loader import Load_Dataset, Dataset_Split
from util.defense import fft_topk_denoise, normalize_iq_data, denormalize_iq_data
from util.detector import RFSignalAutoEncoder, detector_gate_fft_topk


def main():
    parser = argparse.ArgumentParser(description='Test CW attack with recovery')
    parser.add_argument('--dataset', type=str, default='2016.10a')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to AWN classifier checkpoint')
    parser.add_argument('--detector_ckpt', type=str, default=None, help='Path to detector checkpoint (optional)')
    parser.add_argument('--defense', type=str, default='fft_topk', choices=['fft_topk', 'ae_fft_topk'])
    parser.add_argument('--def_topk', type=int, default=50)
    parser.add_argument('--detector_threshold', type=float, default=0.004468)
    parser.add_argument('--cw_steps', type=int, default=100)
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--test_samples', type=int, default=500, help='Number of test samples')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Load config and model
    cfg = Config(args.dataset, train=False)
    cfg = merge_args2cfg(cfg, vars(args))

    model = create_AWN_model(cfg)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"Loaded AWN classifier from {args.ckpt_path}")

    # Load detector if using ae_fft_topk
    detector = None
    if args.defense == 'ae_fft_topk' and args.detector_ckpt:
        detector = RFSignalAutoEncoder().to(device)
        det_ckpt = torch.load(args.detector_ckpt, map_location=device)
        detector.load_state_dict(det_ckpt)
        detector.eval()
        print(f"Loaded detector from {args.detector_ckpt}")

    # Load test data
    print(f"Loading {args.dataset} dataset...")
    Signals, Labels, SNRs, snrs, mods = Load_Dataset(args.dataset, None)
    train_idx, val_idx, test_idx = Dataset_Split(Labels, SNRs, cfg.split_ratio)

    sig_test = Signals[test_idx][:args.test_samples]
    lab_test = Labels[test_idx][:args.test_samples]
    print(f"Testing on {len(sig_test)} samples")

    # Import torchattacks
    try:
        import torchattacks
        print("Using torchattacks library for CW attack")
    except ImportError:
        print("ERROR: torchattacks not installed. Run: pip install torchattacks")
        return

    # Run CW attack
    print(f"\nGenerating CW adversarial examples (steps={args.cw_steps}, c={args.cw_c})...")
    from util.adv_attack import Model01Wrapper, iq_to_ta_input, ta_output_to_iq
    wrapped = Model01Wrapper(model).eval()
    atk = torchattacks.CW(wrapped, c=args.cw_c, kappa=0.0, steps=args.cw_steps, lr=1e-2)

    batch_size = 50
    adv_all = []
    clean_preds = []
    adv_preds = []
    recovered_preds = []

    for i in tqdm(range(0, len(sig_test), batch_size)):
        batch_sig = sig_test[i:i+batch_size].to(device)
        batch_lab = lab_test[i:i+batch_size].to(device)

        # Clean predictions
        with torch.no_grad():
            logits_clean, _ = model(batch_sig)
            pred_clean = torch.argmax(logits_clean, dim=1)
            clean_preds.append(pred_clean.cpu())

        # Generate adversarial examples with proper [0,1]↔[-1,1] mapping
        batch_x01_4d = iq_to_ta_input(batch_sig)
        adv01_4d = atk(batch_x01_4d, batch_lab)
        adv = ta_output_to_iq(adv01_4d)
        adv_all.append(adv.cpu())

        # Adversarial predictions
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
        else:
            # Simple FFT Top-K without detector
            with torch.no_grad():
                adv_norm = normalize_iq_data(adv, 0.02, 0.04)
                adv_filt = fft_topk_denoise(adv_norm, topk=args.def_topk)
                adv_recovered = denormalize_iq_data(adv_filt, 0.02, 0.04)

        # Recovered predictions
        with torch.no_grad():
            logits_rec, _ = model(adv_recovered)
            pred_rec = torch.argmax(logits_rec, dim=1)
            recovered_preds.append(pred_rec.cpu())

    # Compute accuracies
    clean_preds = torch.cat(clean_preds)
    adv_preds = torch.cat(adv_preds)
    recovered_preds = torch.cat(recovered_preds)
    labels = lab_test[:len(clean_preds)].cpu()

    clean_acc = (clean_preds == labels).float().mean().item() * 100
    adv_acc = (adv_preds == labels).float().mean().item() * 100
    recovered_acc = (recovered_preds == labels).float().mean().item() * 100

    print(f"\n{'='*60}")
    print(f"Results on {len(labels)} samples:")
    print(f"  Clean Accuracy:     {clean_acc:.2f}%")
    print(f"  Adversarial Acc:    {adv_acc:.2f}%")
    print(f"  Recovered Acc:      {recovered_acc:.2f}%")
    print(f"  Recovery Gain:      {recovered_acc - adv_acc:+.2f}%")
    print(f"{'='*60}")

    # Save results
    results = {
        'clean_acc': clean_acc,
        'adv_acc': adv_acc,
        'recovered_acc': recovered_acc,
        'defense': args.defense,
        'topk': args.def_topk,
        'cw_steps': args.cw_steps,
        'cw_c': args.cw_c,
    }

    output_path = f"cw_recovery_test_{args.dataset}.pt"
    torch.save(results, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
