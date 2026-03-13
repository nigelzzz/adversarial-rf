#!/usr/bin/env python
"""
CRC Experiment: AMC Accuracy vs Data Integrity under CW Attack.

Two-track design:
  Track A (AMC): Uses real RML2016 signals to measure classification accuracy
                 under clean vs CW adversarial conditions.
  Track B (CRC): Uses synthetic signals with known bits to measure data
                 integrity (CRC pass rate) under correct vs wrong demodulation,
                 and with/without CW perturbation.

Combines both tracks to show four scenarios:
1. Clean + Oracle demod  — baseline, receiver knows true modulation
2. Clean + AMC demod     — receiver uses AWN's prediction (simulated from Track A)
3. Adv + Oracle demod    — CW perturbation applied, correct demod
4. Adv + AMC demod       — CW perturbation applied, AWN picks (wrong) demod

Usage:
    python crc_experiment.py --n_bursts 100 --snr_list 18
    python crc_experiment.py --mod_types BPSK,QPSK --snr_list 0,18 --n_bursts 1000
"""

import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import torch

from util.synth_txrx import (
    generate_burst, demodulate_burst, get_bits_per_symbol,
)
from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial


# AWN class index -> modulation name (all 11 classes)
IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}

# Modulation name -> AWN class index
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}

# Subset that our synthetic TX/RX chain can demodulate (all digital mods)
DEMOD_MODS = {'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4', 'CPFSK', 'GFSK'}

# For backward compat
IDX_TO_DEMOD_MOD = {k: v for k, v in IDX_TO_MOD.items() if v in DEMOD_MODS}


def load_model(ckpt_path, device):
    """Load AWN model from checkpoint."""
    cfg = Config('2016.10a', train=False)
    cfg.device = device
    model = create_model(cfg, model_name='awn')

    ckpt_file = os.path.join(ckpt_path, '2016.10a_AWN.pkl')
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    state_dict = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


def load_rml2016_subset(mod_type, snr, n_samples=None):
    """Load real RML2016 data for a specific (mod, snr)."""
    with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    key = (mod_type.encode() if isinstance(mod_type, str) else mod_type, snr)
    samples = data[key]  # [N, 2, 128]
    if n_samples is not None and n_samples < len(samples):
        idx = np.random.choice(len(samples), n_samples, replace=False)
        samples = samples[idx]
    return samples


@torch.no_grad()
def classify_batch(model, iq_batch_np, device):
    """Run AWN on a numpy IQ batch [N,2,T]. Returns predicted class indices."""
    x = torch.from_numpy(iq_batch_np).float().to(device)
    logits, _ = model(x)
    return logits.argmax(dim=1).cpu().numpy()


def demod_with_iq(iq_complex, burst_info, use_mod_type, adv_iq=None):
    """Demodulate IQ samples using given modulation type and burst params.

    If adv_iq is provided, splices the adversarial 128-sample window into the
    full signal (replacing the middle) before applying the RX matched filter.
    This preserves the guard context for edge-ISI-free demodulation.
    """
    try:
        iq_full = burst_info.get('iq_full')
        iq_win_start = burst_info.get('iq_win_start')
        n_guard = burst_info.get('n_guard')

        # For adversarial signals, splice into the full signal
        if adv_iq is not None and iq_full is not None:
            iq_full = iq_full.copy()
            n_win = len(adv_iq)
            iq_full[iq_win_start:iq_win_start + n_win] = adv_iq
            iq_complex = adv_iq

        result = demodulate_burst(
            iq_complex,
            use_mod_type,
            n_pilots=burst_info['n_pilots'],
            pilot_symbols=burst_info.get('pilot_symbols'),
            pilot_bits=burst_info.get('pilot_bits'),
            sps=8,
            beta=0.35,
            pilot_positions=burst_info.get('pilot_positions'),
            iq_full=iq_full,
            iq_win_start=iq_win_start,
            n_guard=n_guard,
        )
        return result['crc_pass']
    except Exception:
        return False


def run_experiment(args):
    """Main experiment loop."""
    fix_seed(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    attack_name = getattr(args, 'attack', 'cw').lower()

    print(f"Device: {device}")
    print(f"Attack: {attack_name}")
    print(f"Modulations: {args.mod_types}")
    print(f"SNR list: {args.snr_list}")
    print(f"N bursts (synthetic): {args.n_bursts}")
    if attack_name == 'cw':
        print(f"CW params: c={args.cw_c}, steps={args.cw_steps}")
    print()

    # Load model
    print("Loading AWN model...")
    model, cfg = load_model(args.ckpt_path, device)
    cfg.cw_c = args.cw_c
    cfg.cw_steps = args.cw_steps
    cfg.cw_lr = 0.005
    cfg.attack_eps = getattr(args, 'attack_eps', 0.1)
    cfg.ta_box = args.ta_box
    cfg.num_classes = 11

    # Wrap model for torchattacks
    wrapped_model = Model01Wrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # Create attack
    print(f"Creating {attack_name.upper()} attack...")
    attack = create_attack(attack_name, wrapped_model, cfg)

    all_results = []

    for mod_type in args.mod_types:
        for snr_db in args.snr_list:
            print(f"\n{'='*60}")
            print(f"  {mod_type} @ SNR={snr_db} dB")
            print(f"{'='*60}")

            # ==============================================================
            # Track A: AMC accuracy using real RML2016 data
            # ==============================================================
            print("\n  [Track A] AMC on real RML2016 data...")
            real_data = load_rml2016_subset(mod_type, snr_db, n_samples=500)
            true_label_idx = MOD_TO_IDX[mod_type]
            real_labels = np.full(len(real_data), true_label_idx, dtype=np.int64)
            real_labels_t = torch.from_numpy(real_labels).to(device)

            # Clean AMC
            clean_preds = classify_batch(model, real_data, device)
            clean_amc_acc = np.mean(clean_preds == true_label_idx)
            print(f"  Clean AMC accuracy: {100*clean_amc_acc:.1f}%")

            # Adversarial attack on real data
            print(f"  Running {attack_name.upper()} attack on real data...")
            real_t = torch.from_numpy(real_data).float().to(device)
            adv_real_list = []
            bs = args.batch_size
            for i in range(0, len(real_data), bs):
                x_b = real_t[i:i+bs]
                y_b = real_labels_t[i:i+bs]
                x_adv = generate_adversarial(
                    attack, x_b, y_b,
                    wrapped_model=wrapped_model,
                    ta_box=args.ta_box,
                )
                adv_real_list.append(x_adv.cpu())
            adv_real = torch.cat(adv_real_list, dim=0).numpy()

            # Adversarial AMC
            adv_preds = classify_batch(model, adv_real, device)
            adv_amc_acc = np.mean(adv_preds == true_label_idx)
            print(f"  Adversarial AMC accuracy: {100*adv_amc_acc:.1f}%")

            # Adversarial prediction distribution (for simulating AMC demod)
            adv_pred_counts = Counter(adv_preds.tolist())
            print(f"  Adv prediction dist: "
                  f"{[(IDX_TO_MOD.get(k, f'cls{k}'), v) for k, v in adv_pred_counts.most_common(5)]}")

            # Perturbation statistics
            real_delta = adv_real - real_data
            real_l2 = np.sqrt(np.sum(real_delta**2, axis=(1, 2)))
            print(f"  Perturbation L2: mean={np.mean(real_l2):.6f}, "
                  f"max={np.max(real_l2):.6f}")

            # ==============================================================
            # Track B: CRC integrity (only for demodulatable modulations)
            # ==============================================================
            can_demod = mod_type in DEMOD_MODS

            if can_demod:
                print(f"\n  [Track B] CRC on {args.n_bursts} synthetic bursts...")
                rng = np.random.default_rng(args.seed)

                bursts = []
                for _ in range(args.n_bursts):
                    b = generate_burst(
                        mod_type, n_symbols=16, n_pilots=2, sps=8, beta=0.35,
                        snr_db=snr_db, target_rms=0.006, cfo_std=0.0, rng=rng,
                    )
                    bursts.append(b)

                synth_iq_np = np.concatenate(
                    [b['iq_tensor'] for b in bursts], axis=0
                )

                # Transfer CW perturbation from real data to synthetic
                n_real = len(real_delta)
                sample_idx = rng.integers(0, n_real, size=args.n_bursts)
                adv_synth_np = synth_iq_np + real_delta[sample_idx]

                synth_delta = adv_synth_np - synth_iq_np
                l2_synth = np.sqrt(np.sum(synth_delta**2, axis=(1, 2)))
                rms_signal = np.sqrt(np.mean(synth_iq_np**2, axis=(1, 2)))
                snr_perturb = 20 * np.log10(
                    rms_signal / (np.sqrt(np.mean(synth_delta**2, axis=(1, 2))) + 1e-12)
                )
                print(f"  Transferred perturbation L2: mean={np.mean(l2_synth):.6f}, "
                      f"Signal/Perturb SNR: {np.mean(snr_perturb):.1f} dB")

                # Demodulate under 4 scenarios
                print("  Demodulating...")
                adv_pred_list = list(adv_preds)
                n_adv_preds = len(adv_pred_list)

                counts = {
                    'clean_oracle': 0, 'clean_amc': 0,
                    'adv_oracle': 0, 'adv_amc': 0,
                }

                for idx in range(args.n_bursts):
                    burst_info = bursts[idx]
                    clean_iq = burst_info['iq_complex']
                    adv_iq = (adv_synth_np[idx, 0, :]
                              + 1j * adv_synth_np[idx, 1, :])

                    # Scenario 1: Clean + Oracle
                    counts['clean_oracle'] += int(
                        demod_with_iq(clean_iq, burst_info, mod_type))

                    # Scenario 2: Clean + AMC (simulate from Track A dist)
                    if rng.random() < clean_amc_acc:
                        pred_mod = mod_type
                    else:
                        wrong = clean_preds[clean_preds != true_label_idx]
                        sc = int(rng.choice(wrong)) if len(wrong) > 0 else true_label_idx
                        pred_mod = IDX_TO_DEMOD_MOD.get(sc)
                    crc2 = (demod_with_iq(clean_iq, burst_info, pred_mod)
                            if pred_mod else False)
                    counts['clean_amc'] += int(crc2)

                    # Scenario 3: Adversarial + Oracle (splice adv into full)
                    counts['adv_oracle'] += int(
                        demod_with_iq(clean_iq, burst_info, mod_type,
                                      adv_iq=adv_iq))

                    # Scenario 4: Adversarial + AMC
                    sc = int(adv_pred_list[idx % n_adv_preds])
                    apm = IDX_TO_DEMOD_MOD.get(sc)
                    if apm:
                        crc4 = demod_with_iq(clean_iq, burst_info, apm,
                                             adv_iq=adv_iq)
                    else:
                        crc4 = False
                    counts['adv_amc'] += int(crc4)

                n = args.n_bursts
                crc_clean_oracle = counts['clean_oracle'] / n
                crc_clean_amc = counts['clean_amc'] / n
                crc_adv_oracle = counts['adv_oracle'] / n
                crc_adv_amc = counts['adv_amc'] / n
            else:
                print(f"\n  [Track B] Skipped (no digital demod for {mod_type})")
                n = 0
                crc_clean_oracle = crc_clean_amc = None
                crc_adv_oracle = crc_adv_amc = None

            # ==============================================================
            # Combined results
            # ==============================================================
            scenarios = [
                ('Clean+Oracle', clean_amc_acc, crc_clean_oracle),
                ('Clean+AMC', clean_amc_acc, crc_clean_amc),
                ('Adv+Oracle', adv_amc_acc, crc_adv_oracle),
                ('Adv+AMC', adv_amc_acc, crc_adv_amc),
            ]

            for scenario_name, amc_acc, crc_rate in scenarios:
                all_results.append({
                    'mod': mod_type,
                    'snr': snr_db,
                    'scenario': scenario_name,
                    'amc_acc': round(amc_acc, 4),
                    'crc_pass': round(crc_rate, 4) if crc_rate is not None else None,
                    'n': args.n_bursts if can_demod else 0,
                })

            print(f"\n  {'Scenario':<18} {'AMC Acc':>10} {'CRC Pass':>10}")
            print(f"  {'-'*40}")
            for scenario_name, amc_acc, crc_rate in scenarios:
                crc_str = f"{100*crc_rate:>9.1f}%" if crc_rate is not None else "      N/A"
                print(f"  {scenario_name:<18} {100*amc_acc:>9.1f}% {crc_str}")

    # --- Output results ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Console table
    print(f"\n\n{'='*72}")
    print(f"  CRC Experiment Results — {attack_name.upper()} Attack")
    print(f"  (Track A: RML2016 AMC, Track B: Synthetic CRC)")
    print(f"{'='*72}")
    print(f"  {'Mod':<8} {'SNR':>5} {'Scenario':<18} {'AMC Acc':>10} {'CRC Pass':>10} {'N':>6}")
    print(f"  {'-'*62}")
    for r in all_results:
        crc_str = f"{100*r['crc_pass']:>9.1f}%" if r['crc_pass'] is not None else "      N/A"
        print(f"  {r['mod']:<8} {r['snr']:>5} {r['scenario']:<18} "
              f"{100*r['amc_acc']:>9.1f}% {crc_str} {r['n']:>6}")
    print(f"{'='*72}")

    # CSV
    csv_path = os.path.join(args.output_dir, 'crc_vs_amc.csv')
    with open(csv_path, 'w') as f:
        f.write('mod,snr,scenario,amc_acc,crc_pass,n\n')
        for r in all_results:
            f.write(f"{r['mod']},{r['snr']},{r['scenario']},"
                    f"{r['amc_acc']},{r['crc_pass']},{r['n']}\n")
    print(f"\nCSV saved to: {csv_path}")

    # JSON
    json_path = os.path.join(args.output_dir, 'crc_vs_amc.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON saved to: {json_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='CRC Experiment: AMC Accuracy vs Data Integrity under CW Attack'
    )
    parser.add_argument('--attack', type=str, default='cw',
                        help='Attack type: cw, deepfool, eadl1, eaden, fab, fgsm, pgd')
    parser.add_argument('--attack_eps', type=float, default=0.1,
                        help='Epsilon for Linf/boundary attacks (default 0.1)')
    parser.add_argument('--mod_types', type=str,
                        default='BPSK,QPSK,8PSK,QAM16,QAM64,PAM4,CPFSK,GFSK,WBFM,AM-DSB,AM-SSB',
                        help='Comma-separated modulation types')
    parser.add_argument('--snr_list', type=str, default='0,18',
                        help='Comma-separated SNR values in dB')
    parser.add_argument('--n_bursts', type=int, default=1000,
                        help='Number of synthetic bursts per (mod, snr) cell')
    parser.add_argument('--cw_c', type=float, default=10.0,
                        help='CW attack confidence weight')
    parser.add_argument('--cw_steps', type=int, default=200,
                        help='CW optimization steps')
    parser.add_argument('--ta_box', type=str, default='minmax',
                        choices=['unit', 'minmax', 'paper'],
                        help='Normalization mode for torchattacks')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint',
                        help='Model checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='./crc_experiment_results',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Attack batch size')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, or cuda')

    args = parser.parse_args()

    # Parse comma-separated lists
    args.mod_types = [m.strip() for m in args.mod_types.split(',')]
    args.snr_list = [int(s.strip()) for s in args.snr_list.split(',')]

    run_experiment(args)


if __name__ == '__main__':
    main()
