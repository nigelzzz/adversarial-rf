#!/usr/bin/env python
"""
Burst Length Ablation: CRC vs frame length under CW adversarial attack.

Answers reviewer question: "How sensitive are your CRC results to burst length?"

Methodology (same Track A/B split as main pipeline):
  Track A: Run CW attack on REAL RML2016.10a data → get AMC predictions + deltas
  Track B: Generate SYNTHETIC bursts at different lengths, transfer deltas,
           demodulate with oracle vs AMC-driven demod, check CRC

Key insight:
  - AWN always sees 128 samples → attack effectiveness on AMC is constant
  - But the FRAME length varies (16/64/128 symbols) → CRC covers more bits
  - Wrong demod → ~50% BER on ALL symbols → CRC fails regardless of length
  - Right demod with perturbation → only 16/N symbols affected

Usage:
    python burst_length_experiment.py --snr 18
    python burst_length_experiment.py --snr 14 --n_bursts 500
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util.synth_txrx import (
    generate_burst, demodulate_burst, get_bits_per_symbol,
    ALL_DIGITAL_MODS, CONSTELLATION_MODS, FSK_MODS, crc8_check,
)
from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial

# ============================================================
# Constants
# ============================================================

IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}

AWN_WINDOW = 128  # AWN input: 128 samples = 16 symbols at sps=8


# ============================================================
# Helpers
# ============================================================

def load_rml2016(path, mod_type, snr, n_samples=None, rng=None):
    """Load real RML2016.10a samples for (mod, snr)."""
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    key = (mod_type.encode() if isinstance(mod_type, str) else mod_type, snr)
    samples = data[key]
    if n_samples is not None and n_samples < len(samples):
        idx = rng.choice(len(samples), n_samples, replace=False) if rng else \
              np.random.choice(len(samples), n_samples, replace=False)
        samples = samples[idx]
    return samples


@torch.no_grad()
def classify_batch(model, iq_np, device, batch_size=256):
    preds = []
    for i in range(0, len(iq_np), batch_size):
        x = torch.from_numpy(iq_np[i:i+batch_size]).float().to(device)
        logits, _ = model(x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def attack_batch(attack, x_np, labels_np, wrapped_model, device,
                 ta_box='minmax', batch_size=128):
    x_t = torch.from_numpy(x_np).float().to(device)
    labels_t = torch.from_numpy(labels_np).long().to(device)
    adv_list = []
    for i in range(0, len(x_np), batch_size):
        x_b = x_t[i:i+batch_size]
        y_b = labels_t[i:i+batch_size]
        x_adv = generate_adversarial(attack, x_b, y_b,
                                     wrapped_model=wrapped_model,
                                     ta_box=ta_box)
        adv_list.append(x_adv.cpu())
    return torch.cat(adv_list, dim=0).numpy()


def demod_with_iq(iq_complex, burst_info, use_mod_type, adv_iq=None):
    """Demodulate burst, optionally splicing adversarial IQ window."""
    try:
        iq_full = burst_info.get('iq_full')
        iq_win_start = burst_info.get('iq_win_start')
        n_guard = burst_info.get('n_guard')

        if adv_iq is not None and iq_full is not None:
            iq_full = iq_full.copy()
            n_win = len(adv_iq)
            awn_offset = burst_info.get('awn_offset', 0)
            splice_start = iq_win_start + awn_offset
            iq_full[splice_start:splice_start + n_win] = adv_iq
            iq_complex = iq_complex.copy()
            iq_complex[awn_offset:awn_offset + n_win] = adv_iq

        result = demodulate_burst(
            iq_complex, use_mod_type,
            n_pilots=burst_info['n_pilots'],
            pilot_symbols=burst_info.get('pilot_symbols'),
            pilot_bits=burst_info.get('pilot_bits'),
            sps=8, beta=0.35,
            pilot_positions=burst_info.get('pilot_positions'),
            iq_full=iq_full, iq_win_start=iq_win_start, n_guard=n_guard,
        )
        return result
    except Exception:
        bps = get_bits_per_symbol(use_mod_type) if use_mod_type in ALL_DIGITAL_MODS else 1
        n_frame = 14 * bps
        return {
            'crc_pass': False,
            'recovered_bits': np.zeros(n_frame, dtype=np.uint8),
            'data_bits': np.zeros(max(n_frame - 8, 0), dtype=np.uint8),
            'recovered_symbols': None,
        }


def compute_ber(tx_bits, rx_bits):
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return 0.0
    return float(np.sum(tx_bits[:n] != rx_bits[:n])) / n


# ============================================================
# Main experiment
# ============================================================

def run_experiment(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    os.makedirs(args.output_dir, exist_ok=True)
    fix_seed(args.seed)

    # Load AWN model
    cfg = Config('2016.10a', train=False)
    cfg.device = device
    model = create_model(cfg, model_name='awn')
    ckpt_file = os.path.join(args.ckpt, '2016.10a_AWN.pkl')
    model.load_state_dict(torch.load(ckpt_file, map_location=device))
    model.eval()

    wrapped_model = Model01Wrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # Create CW attack
    cfg.cw_c = args.cw_c
    cfg.cw_steps = args.cw_steps
    cfg.cw_lr = 0.01
    cfg.attack_eps = 0.03
    cfg.ta_box = 'minmax'
    cw_attack = create_attack('cw', wrapped_model, cfg)

    snr = args.snr
    n_bursts = args.n_bursts
    mods = args.mods.split(',')
    burst_lengths = [int(x) for x in args.burst_lengths.split(',')]

    print("=" * 75)
    print("  Burst Length Ablation: CRC vs Frame Length under CW Attack")
    print("=" * 75)
    print(f"  SNR:           {snr} dB")
    print(f"  Burst lengths: {burst_lengths} symbols")
    print(f"  Mods:          {mods}")
    print(f"  N bursts:      {n_bursts}")
    print(f"  CW:            c={args.cw_c}, steps={args.cw_steps}")
    print(f"  Device:        {device}")
    print(f"  Approach:      Track A (real RML) + Track B (synth, var length)")

    rng_global = np.random.default_rng(args.seed)

    # ================================================================
    # Track A: Real RML data — AMC + CW attack (once per mod)
    # ================================================================
    print(f"\n{'='*75}")
    print(f"  Track A: AMC + CW on real RML2016.10a (SNR={snr})")
    print(f"{'='*75}")

    track_a = {}  # mod -> {clean_acc, adv_preds, delta, ...}
    n_real = 500

    for mod in mods:
        if mod not in ALL_DIGITAL_MODS:
            continue

        real_data = load_rml2016(args.dataset, mod, snr, n_real, rng_global)
        true_idx = MOD_TO_IDX[mod]

        # Clean AMC
        clean_preds = classify_batch(model, real_data, device)
        clean_acc = float(np.mean(clean_preds == true_idx))

        # CW attack
        labels = np.full(len(real_data), true_idx)
        adv_data = attack_batch(cw_attack, real_data, labels,
                                wrapped_model, device, ta_box='minmax')
        adv_preds = classify_batch(model, adv_data, device)
        adv_acc = float(np.mean(adv_preds == true_idx))
        delta = adv_data - real_data  # [N, 2, 128] perturbation

        track_a[mod] = {
            'clean_acc': clean_acc,
            'clean_preds': clean_preds,
            'adv_acc': adv_acc,
            'adv_preds': adv_preds,
            'delta': delta,
            'true_idx': true_idx,
        }

        l2 = np.sqrt(np.sum(delta**2, axis=(1, 2)))
        print(f"  {mod:6s}: Clean AMC={100*clean_acc:5.1f}%  "
              f"Adv AMC={100*adv_acc:5.1f}%  "
              f"delta L2={np.mean(l2):.5f}")

    # ================================================================
    # Track B: Synthetic bursts at variable lengths
    # ================================================================
    all_rows = []

    for n_sym in burst_lengths:
        sps = 8
        n_samples = n_sym * sps
        awn_offset = max(0, (n_samples - AWN_WINDOW) // 2)

        print(f"\n{'='*75}")
        print(f"  Track B: {n_sym} symbols ({n_samples} samples), "
              f"AWN window [{awn_offset}:{awn_offset+AWN_WINDOW}]")
        print(f"{'='*75}")

        for mod in mods:
            if mod not in ALL_DIGITAL_MODS:
                continue

            ta = track_a[mod]
            t0 = time.time()
            rng = np.random.default_rng(args.seed + hash((mod, snr, n_sym)) % 2**31)

            # Generate synthetic bursts
            bursts = []
            for _ in range(n_bursts):
                b = generate_burst(mod, n_symbols=n_sym, n_pilots=2, sps=sps,
                                   beta=0.35, snr_db=snr, target_rms=0.006,
                                   cfo_std=0.0, rng=rng, n_guard=16)
                b['awn_offset'] = awn_offset
                bursts.append(b)

            # Sample perturbation deltas from Track A
            n_real_deltas = len(ta['delta'])
            delta_idx = rng.integers(0, n_real_deltas, size=n_bursts)

            # Build adversarial IQ windows (128-sample perturbation on
            # the AWN window region of each synthetic burst)
            adv_iqs = []
            for idx in range(n_bursts):
                bi = bursts[idx]
                # Extract the 128-sample window from synthetic burst
                synth_window = bi['iq_complex'][awn_offset:awn_offset + AWN_WINDOW]
                # Add perturbation delta (from real RML attack)
                d = ta['delta'][delta_idx[idx]]
                d_complex = d[0, :] + 1j * d[1, :]
                adv_window = synth_window + d_complex
                adv_iqs.append(adv_window)

            # Use Track A predictions for AMC-driven demod
            adv_preds = ta['adv_preds']
            clean_preds = ta['clean_preds']

            bps = get_bits_per_symbol(mod)
            n_data_sym = n_sym - 2
            n_frame_bits = n_data_sym * bps

            # Evaluate 4 scenarios
            scenarios = {
                'Clean+Oracle': {'crc': 0, 'ber': []},
                'Clean+AMC':    {'crc': 0, 'ber': []},
                'Adv+Oracle':   {'crc': 0, 'ber': []},
                'Adv+AMC':      {'crc': 0, 'ber': []},
            }

            for idx in range(n_bursts):
                bi = bursts[idx]
                clean_iq = bi['iq_complex']
                adv_iq = adv_iqs[idx]

                # --- Clean + Oracle ---
                r1 = demod_with_iq(clean_iq, bi, mod)
                scenarios['Clean+Oracle']['crc'] += int(r1['crc_pass'])
                scenarios['Clean+Oracle']['ber'].append(
                    compute_ber(bi['frame_bits'], r1['recovered_bits']))

                # --- Clean + AMC (using Track A clean prediction) ---
                cp = int(clean_preds[idx % len(clean_preds)])
                cp_mod = IDX_TO_MOD.get(cp)
                if cp_mod and cp_mod in ALL_DIGITAL_MODS:
                    r2 = demod_with_iq(clean_iq, bi, cp_mod)
                else:
                    r2 = {'crc_pass': False, 'recovered_bits': np.array([])}
                scenarios['Clean+AMC']['crc'] += int(r2['crc_pass'])
                scenarios['Clean+AMC']['ber'].append(
                    compute_ber(bi['frame_bits'], r2.get('recovered_bits', np.array([]))))

                # --- Adv + Oracle ---
                r3 = demod_with_iq(clean_iq, bi, mod, adv_iq=adv_iq)
                scenarios['Adv+Oracle']['crc'] += int(r3['crc_pass'])
                scenarios['Adv+Oracle']['ber'].append(
                    compute_ber(bi['frame_bits'], r3['recovered_bits']))

                # --- Adv + AMC (using Track A adv prediction) ---
                ap = int(adv_preds[idx % len(adv_preds)])
                ap_mod = IDX_TO_MOD.get(ap)
                if ap_mod and ap_mod in ALL_DIGITAL_MODS:
                    r4 = demod_with_iq(clean_iq, bi, ap_mod, adv_iq=adv_iq)
                else:
                    r4 = {'crc_pass': False, 'recovered_bits': np.array([])}
                scenarios['Adv+AMC']['crc'] += int(r4['crc_pass'])
                scenarios['Adv+AMC']['ber'].append(
                    compute_ber(bi['frame_bits'], r4.get('recovered_bits', np.array([]))))

            elapsed = time.time() - t0
            print(f"\n  {mod} @ {n_sym} sym ({n_frame_bits} frame bits) [{elapsed:.1f}s]")
            print(f"    Track A: Clean AMC={100*ta['clean_acc']:.1f}%  "
                  f"Adv AMC={100*ta['adv_acc']:.1f}%")
            print(f"    {'Scenario':<16} {'CRC':>8} {'BER':>10}")
            print(f"    {'-'*36}")

            for sc_name in ['Clean+Oracle', 'Clean+AMC', 'Adv+Oracle', 'Adv+AMC']:
                sc = scenarios[sc_name]
                crc_rate = sc['crc'] / n_bursts
                mean_ber = float(np.mean(sc['ber']))
                print(f"    {sc_name:<16} {100*crc_rate:>7.1f}% {mean_ber:>10.5f}")

                all_rows.append({
                    'mod': mod, 'snr': snr, 'n_symbols': n_sym,
                    'n_frame_bits': n_frame_bits,
                    'scenario': sc_name,
                    'crc_pass_rate': round(crc_rate, 4),
                    'ber': round(mean_ber, 6),
                    'clean_amc_acc': round(ta['clean_acc'], 4),
                    'adv_amc_acc': round(ta['adv_acc'], 4),
                })

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'burst_length_ablation.csv')
    cols = ['mod', 'snr', 'n_symbols', 'n_frame_bits', 'scenario',
            'crc_pass_rate', 'ber', 'clean_amc_acc', 'adv_amc_acc']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in all_rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f"\n  Saved: {csv_path}")

    # Save JSON
    json_path = os.path.join(args.output_dir, 'burst_length_ablation.json')
    with open(json_path, 'w') as f:
        json.dump(all_rows, f, indent=2)
    print(f"  Saved: {json_path}")

    # Plot
    plot_ablation(all_rows, burst_lengths, mods, snr, args.output_dir)

    # Print summary table
    print_summary(all_rows, burst_lengths, mods, snr)

    return all_rows


def print_summary(rows, burst_lengths, mods, snr):
    """Print compact summary table."""
    digital = [m for m in mods if m in ALL_DIGITAL_MODS]

    print(f"\n{'='*75}")
    print(f"  SUMMARY: Burst Length Ablation (SNR={snr} dB, CW attack)")
    print(f"{'='*75}")

    # Header
    header = f"  {'Mod':<6}"
    for bl in burst_lengths:
        header += f" | {bl:>3}sym CO  CA  AO  AA"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for mod in digital:
        line = f"  {mod:<6}"
        for bl in burst_lengths:
            vals = {}
            for sc in ['Clean+Oracle', 'Clean+AMC', 'Adv+Oracle', 'Adv+AMC']:
                short = {'Clean+Oracle':'CO', 'Clean+AMC':'CA',
                         'Adv+Oracle':'AO', 'Adv+AMC':'AA'}[sc]
                r = [x for x in rows if x['mod']==mod and x['n_symbols']==bl
                     and x['scenario']==sc]
                vals[short] = r[0]['crc_pass_rate'] * 100 if r else 0
            line += f" | {vals['CO']:>5.0f} {vals['CA']:>3.0f} {vals['AO']:>3.0f} {vals['AA']:>3.0f}"
        print(line)

    print(f"\n  CO=Clean+Oracle  CA=Clean+AMC  AO=Adv+Oracle  AA=Adv+AMC")
    print(f"  Values are CRC pass rate (%)")


def plot_ablation(rows, burst_lengths, mods, snr, output_dir):
    """Generate burst-length ablation figures."""
    digital_mods = [m for m in mods if m in ALL_DIGITAL_MODS]
    colors = plt.cm.tab10(np.linspace(0, 1, len(digital_mods)))

    # Figure 1: CRC pass rate by burst length for each scenario
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    scenario_names = ['Clean+Oracle', 'Clean+AMC', 'Adv+Oracle', 'Adv+AMC']

    for ax, sc_name in zip(axes, scenario_names):
        for mi, mod in enumerate(digital_mods):
            pts = [(r['n_symbols'], r['crc_pass_rate'] * 100)
                   for r in rows if r['mod'] == mod and r['scenario'] == sc_name]
            if pts:
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        'o-', color=colors[mi], label=mod, linewidth=2, markersize=8)
        ax.set_xlabel('Burst Length (symbols)')
        ax.set_title(sc_name, fontsize=12, fontweight='bold')
        ax.set_xticks(burst_lengths)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('CRC Pass Rate (%)')
        if ax == axes[-1]:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    plt.suptitle(f'CRC vs Burst Length under CW Attack (SNR={snr} dB)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'fig_burst_length_ablation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Figure 2: Control-plane gap by burst length
    fig, ax = plt.subplots(figsize=(10, 6))
    for mi, mod in enumerate(digital_mods):
        gaps = []
        for bl in burst_lengths:
            ao = [r['crc_pass_rate'] for r in rows
                  if r['mod'] == mod and r['n_symbols'] == bl
                  and r['scenario'] == 'Adv+Oracle']
            aa = [r['crc_pass_rate'] for r in rows
                  if r['mod'] == mod and r['n_symbols'] == bl
                  and r['scenario'] == 'Adv+AMC']
            if ao and aa:
                gaps.append((bl, (ao[0] - aa[0]) * 100))
        if gaps:
            gaps.sort()
            ax.plot([g[0] for g in gaps], [g[1] for g in gaps],
                    'o-', color=colors[mi], label=mod, linewidth=2, markersize=8)

    ax.set_xlabel('Burst Length (symbols)')
    ax.set_ylabel('Control-Plane Gap (pp)\n(Adv+Oracle CRC - Adv+AMC CRC)')
    ax.set_title(f'Control-Plane Attack Gap vs Burst Length (SNR={snr} dB)')
    ax.set_xticks(burst_lengths)
    ax.set_ylim(-5, 105)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'fig_burst_length_gap.png')
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path2}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Burst Length Ablation')
    parser.add_argument('--snr', type=int, default=18)
    parser.add_argument('--n_bursts', type=int, default=200)
    parser.add_argument('--burst_lengths', type=str, default='16,64,128')
    parser.add_argument('--mods', type=str,
                        default='BPSK,QPSK,8PSK,QAM16,QAM64,PAM4')
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_steps', type=int, default=1000)
    parser.add_argument('--ckpt', type=str, default='./checkpoint')
    parser.add_argument('--dataset', type=str,
                        default='./data/RML2016.10a_dict.pkl')
    parser.add_argument('--output_dir', type=str,
                        default='./results/burst_length_ablation')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_experiment(args)
