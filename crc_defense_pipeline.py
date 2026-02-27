#!/usr/bin/env python
"""
Direct End-to-End CRC Defense Pipeline:
    synth(CRC) → IQ → AWN → CW attack → FFT Top-K → AWN → demod → CRC check

Generates synthetic bursts with CRC at target_rms=0.0082 (matching RML2016.10a
amplitude), so AWN classifies them directly — no Track A/B split needed.

Scenarios compared:
  1. Clean + Oracle:      synth IQ → demod(true mod) → CRC
  2. Clean + AMC:         synth IQ → AWN → demod(AWN pred) → CRC
  3. CW + Oracle:         synth IQ → CW → demod(true mod) → CRC
  4. CW + AMC:            synth IQ → CW → AWN → demod(AWN pred) → CRC
  5. CW + Top-K + Oracle: synth IQ → CW → Top-K → demod(true mod) → CRC
  6. CW + Top-K + AMC:    synth IQ → CW → Top-K → AWN → demod(AWN pred) → CRC
  7. Clean + Top-K:       synth IQ → Top-K → demod(true mod) → CRC  (defense cost)

Usage:
    python crc_defense_pipeline.py --snr 18
    python crc_defense_pipeline.py --snr 18 --topk 10,20,50
    python crc_defense_pipeline.py --snr 14 --n_bursts 500 --topk 5,10,20,50
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util.synth_txrx import (
    generate_burst, demodulate_burst, get_bits_per_symbol,
    ALL_DIGITAL_MODS, CONSTELLATION_MODS, FSK_MODS,
)
from util.defense import fft_topk_denoise
from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial

# ============================================================
IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}

# Default target RMS matching RML2016.10a amplitude
DEFAULT_TARGET_RMS = 0.0082


# ============================================================
# Helpers
# ============================================================

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


def apply_fft_topk_np(iq_complex, topk):
    """Apply FFT Top-K to a single complex IQ signal (numpy)."""
    n = len(iq_complex)
    x_tensor = torch.zeros(1, 2, n)
    x_tensor[0, 0, :] = torch.from_numpy(iq_complex.real.astype(np.float32))
    x_tensor[0, 1, :] = torch.from_numpy(iq_complex.imag.astype(np.float32))
    x_filt = fft_topk_denoise(x_tensor, topk=topk)
    return x_filt[0, 0, :].numpy() + 1j * x_filt[0, 1, :].numpy()


def apply_fft_topk_batch(iq_batch_np, topk):
    """Apply FFT Top-K to a batch [N, 2, 128] numpy array."""
    x_t = torch.from_numpy(iq_batch_np).float()
    x_filt = fft_topk_denoise(x_t, topk=topk)
    return x_filt.numpy()


def tensor_to_complex(iq_tensor_2d):
    """Convert [2, T] tensor to complex array."""
    return iq_tensor_2d[0, :] + 1j * iq_tensor_2d[1, :]


def complex_to_tensor(iq_complex):
    """Convert complex array to [2, T] tensor."""
    n = len(iq_complex)
    out = np.zeros((2, n), dtype=np.float32)
    out[0, :] = iq_complex.real
    out[1, :] = iq_complex.imag
    return out


def demod_burst(burst_info, use_mod_type, override_iq_complex=None):
    """Demodulate burst, optionally replacing the IQ with adversarial/recovered."""
    try:
        iq_full = burst_info.get('iq_full')
        iq_win_start = burst_info.get('iq_win_start')
        n_guard = burst_info.get('n_guard')

        if override_iq_complex is not None and iq_full is not None:
            # Splice override IQ into the full signal at the window position
            iq_full = iq_full.copy()
            iq_full[iq_win_start:iq_win_start + len(override_iq_complex)] = \
                override_iq_complex
            iq_complex = override_iq_complex
        else:
            iq_complex = burst_info['iq_complex']

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
        return {'crc_pass': False, 'recovered_bits': np.array([], dtype=np.uint8)}


def compute_ber(tx_bits, rx_bits):
    n = min(len(tx_bits), len(rx_bits))
    if n == 0:
        return 0.5
    return float(np.sum(tx_bits[:n] != rx_bits[:n])) / n


# ============================================================
# Main
# ============================================================

def run(args):
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    os.makedirs(args.output_dir, exist_ok=True)
    fix_seed(args.seed)

    # Load AWN
    cfg = Config('2016.10a', train=False)
    cfg.device = device
    model = create_model(cfg, model_name='awn')
    ckpt_file = os.path.join(args.ckpt, '2016.10a_AWN.pkl')
    model.load_state_dict(torch.load(ckpt_file, map_location=device))
    model.eval()

    wrapped_model = Model01Wrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # CW attack
    cfg.cw_c = args.cw_c
    cfg.cw_steps = args.cw_steps
    cfg.cw_lr = 0.01
    cfg.attack_eps = 0.03
    cfg.ta_box = 'minmax'
    cw_attack = create_attack('cw', wrapped_model, cfg)

    snr = args.snr
    n_bursts = args.n_bursts
    topk_values = [int(x) for x in args.topk.split(',')]
    mods = args.mods.split(',')
    target_rms = args.target_rms

    print("=" * 75)
    print("  Direct End-to-End CRC Defense Pipeline")
    print("  synth(CRC) → IQ → AWN → CW → FFT Top-K → AWN → demod → CRC")
    print("=" * 75)
    print(f"  SNR:         {snr} dB")
    print(f"  Top-K:       {topk_values}")
    print(f"  Mods:        {mods}")
    print(f"  N bursts:    {n_bursts}")
    print(f"  Target RMS:  {target_rms}")
    print(f"  CW:          c={args.cw_c}, steps={args.cw_steps}")
    print(f"  Device:      {device}")

    all_rows = []

    for mod in mods:
        if mod not in ALL_DIGITAL_MODS:
            print(f"\n  Skipping {mod} (not a digital mod)")
            continue

        true_idx = MOD_TO_IDX[mod]
        rng = np.random.default_rng(args.seed + hash((mod, snr)) % 2**31)

        print(f"\n{'='*75}")
        print(f"  {mod} (SNR={snr} dB)")
        print(f"{'='*75}")

        # ---- Step 1: Generate synthetic bursts with CRC ----
        t0 = time.time()
        bursts = []
        for _ in range(n_bursts):
            b = generate_burst(mod, n_symbols=16, n_pilots=2, sps=8,
                               beta=0.35, snr_db=snr, target_rms=target_rms,
                               cfo_std=0.0, rng=rng, n_guard=16)
            bursts.append(b)

        # Stack IQ tensors for batch processing
        iq_batch = np.concatenate([b['iq_tensor'] for b in bursts], axis=0)
        print(f"  Generated {n_bursts} bursts ({time.time()-t0:.1f}s)")
        print(f"  IQ batch shape: {iq_batch.shape}, "
              f"RMS: {np.sqrt(np.mean(iq_batch**2)):.6f}")

        # ---- Step 2: AWN on clean synthetic signals ----
        clean_preds = classify_batch(model, iq_batch, device)
        clean_acc = float(np.mean(clean_preds == true_idx))
        print(f"  Clean AMC accuracy: {100*clean_acc:.1f}%")

        # ---- Step 3: CW attack on synthetic signals ----
        t0 = time.time()
        labels = np.full(n_bursts, true_idx)
        adv_batch = attack_batch(cw_attack, iq_batch, labels,
                                 wrapped_model, device, ta_box='minmax')
        adv_preds = classify_batch(model, adv_batch, device)
        adv_acc = float(np.mean(adv_preds == true_idx))
        print(f"  CW attack ({time.time()-t0:.1f}s): "
              f"AMC accuracy: {100*adv_acc:.1f}%")

        # ---- Step 4: FFT Top-K recovery ----
        topk_recovered = {}
        topk_preds = {}
        topk_accs = {}
        for k in topk_values:
            rec = apply_fft_topk_batch(adv_batch, topk=k)
            rec_p = classify_batch(model, rec, device)
            rec_acc = float(np.mean(rec_p == true_idx))
            topk_recovered[k] = rec
            topk_preds[k] = rec_p
            topk_accs[k] = rec_acc
            print(f"  Top-{k:2d} recovery: AMC accuracy: {100*rec_acc:.1f}%")

        # ---- Step 5: Demodulate under all scenarios and check CRC ----
        bps = get_bits_per_symbol(mod)
        n_frame_bits = (16 - 2) * bps
        scenario_results = {}

        # --- 5a. Clean + Oracle ---
        crc_cnt, ber_sum = 0, 0.0
        for idx in range(n_bursts):
            r = demod_burst(bursts[idx], mod)
            crc_cnt += int(r['crc_pass'])
            ber_sum += compute_ber(bursts[idx]['frame_bits'],
                                   r['recovered_bits'])
        scenario_results['Clean'] = {
            'crc': crc_cnt / n_bursts, 'ber': ber_sum / n_bursts,
            'amc_acc': clean_acc}

        # --- 5b. Clean + AMC ---
        crc_cnt, ber_sum = 0, 0.0
        for idx in range(n_bursts):
            cp = int(clean_preds[idx])
            cp_mod = IDX_TO_MOD.get(cp)
            if cp_mod and cp_mod in ALL_DIGITAL_MODS:
                r = demod_burst(bursts[idx], cp_mod)
            else:
                r = {'crc_pass': False, 'recovered_bits': np.array([])}
            crc_cnt += int(r['crc_pass'])
            ber_sum += compute_ber(bursts[idx]['frame_bits'],
                                   r.get('recovered_bits', np.array([])))
        scenario_results['Clean+AMC'] = {
            'crc': crc_cnt / n_bursts, 'ber': ber_sum / n_bursts,
            'amc_acc': clean_acc}

        # --- 5c. CW + Oracle ---
        crc_cnt, ber_sum = 0, 0.0
        for idx in range(n_bursts):
            adv_iq = tensor_to_complex(adv_batch[idx])
            r = demod_burst(bursts[idx], mod, override_iq_complex=adv_iq)
            crc_cnt += int(r['crc_pass'])
            ber_sum += compute_ber(bursts[idx]['frame_bits'],
                                   r['recovered_bits'])
        scenario_results['CW'] = {
            'crc': crc_cnt / n_bursts, 'ber': ber_sum / n_bursts,
            'amc_acc': adv_acc}

        # --- 5d. CW + AMC ---
        crc_cnt, ber_sum = 0, 0.0
        for idx in range(n_bursts):
            ap = int(adv_preds[idx])
            ap_mod = IDX_TO_MOD.get(ap)
            adv_iq = tensor_to_complex(adv_batch[idx])
            if ap_mod and ap_mod in ALL_DIGITAL_MODS:
                r = demod_burst(bursts[idx], ap_mod,
                                override_iq_complex=adv_iq)
            else:
                r = {'crc_pass': False, 'recovered_bits': np.array([])}
            crc_cnt += int(r['crc_pass'])
            ber_sum += compute_ber(bursts[idx]['frame_bits'],
                                   r.get('recovered_bits', np.array([])))
        scenario_results['CW+AMC'] = {
            'crc': crc_cnt / n_bursts, 'ber': ber_sum / n_bursts,
            'amc_acc': adv_acc}

        # --- 5e. For each Top-K value ---
        for k in topk_values:
            rec_batch = topk_recovered[k]
            rec_p = topk_preds[k]
            rec_acc = topk_accs[k]

            # CW + Top-K + Oracle
            crc_cnt, ber_sum = 0, 0.0
            for idx in range(n_bursts):
                rec_iq = tensor_to_complex(rec_batch[idx])
                r = demod_burst(bursts[idx], mod,
                                override_iq_complex=rec_iq)
                crc_cnt += int(r['crc_pass'])
                ber_sum += compute_ber(bursts[idx]['frame_bits'],
                                       r['recovered_bits'])
            scenario_results[f'CW+Top{k}'] = {
                'crc': crc_cnt / n_bursts, 'ber': ber_sum / n_bursts,
                'amc_acc': rec_acc}

            # CW + Top-K + AMC
            crc_cnt, ber_sum = 0, 0.0
            for idx in range(n_bursts):
                rec_iq = tensor_to_complex(rec_batch[idx])
                rp = int(rec_p[idx])
                rp_mod = IDX_TO_MOD.get(rp)
                if rp_mod and rp_mod in ALL_DIGITAL_MODS:
                    r = demod_burst(bursts[idx], rp_mod,
                                    override_iq_complex=rec_iq)
                else:
                    r = {'crc_pass': False, 'recovered_bits': np.array([])}
                crc_cnt += int(r['crc_pass'])
                ber_sum += compute_ber(bursts[idx]['frame_bits'],
                                       r.get('recovered_bits', np.array([])))
            scenario_results[f'CW+Top{k}+AMC'] = {
                'crc': crc_cnt / n_bursts, 'ber': ber_sum / n_bursts,
                'amc_acc': rec_acc}

            # Clean + Top-K + Oracle (defense cost on clean signal)
            clean_topk = apply_fft_topk_batch(iq_batch, topk=k)
            crc_cnt, ber_sum = 0, 0.0
            for idx in range(n_bursts):
                filt_iq = tensor_to_complex(clean_topk[idx])
                r = demod_burst(bursts[idx], mod,
                                override_iq_complex=filt_iq)
                crc_cnt += int(r['crc_pass'])
                ber_sum += compute_ber(bursts[idx]['frame_bits'],
                                       r['recovered_bits'])
            scenario_results[f'Clean+Top{k}'] = {
                'crc': crc_cnt / n_bursts, 'ber': ber_sum / n_bursts}

        # ---- Print results ----
        print(f"\n  {'Scenario':<22} {'CRC':>8} {'BER':>10}  {'AMC Acc':>8}")
        print(f"  {'-'*52}")

        for sc_name, sc_data in scenario_results.items():
            amc_str = (f"{100*sc_data['amc_acc']:.1f}%"
                       if 'amc_acc' in sc_data else "")
            print(f"  {sc_name:<22} {100*sc_data['crc']:>7.1f}% "
                  f"{sc_data['ber']:>10.5f}  {amc_str:>8}")

            all_rows.append({
                'mod': mod, 'snr': snr, 'scenario': sc_name,
                'crc_pass_rate': round(sc_data['crc'], 4),
                'ber': round(sc_data['ber'], 6),
                'amc_acc': round(sc_data.get('amc_acc', -1), 4),
            })

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'crc_defense_pipeline.csv')
    cols = ['mod', 'snr', 'scenario', 'crc_pass_rate', 'ber', 'amc_acc']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in all_rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f"\n  Saved: {csv_path}")

    # Save JSON
    json_path = os.path.join(args.output_dir, 'crc_defense_pipeline.json')
    with open(json_path, 'w') as f:
        json.dump(all_rows, f, indent=2)
    print(f"  Saved: {json_path}")

    # Plot
    plot_results(all_rows, mods, topk_values, snr, args.output_dir)
    print_summary_table(all_rows, mods, topk_values, snr)

    return all_rows


def print_summary_table(rows, mods, topk_values, snr):
    """Print compact summary."""
    digital = [m for m in mods if m in ALL_DIGITAL_MODS]

    print(f"\n{'='*75}")
    print(f"  SUMMARY: Direct End-to-End CRC Defense Pipeline (SNR={snr} dB)")
    print(f"  synth(CRC) → IQ → AWN → CW → FFT Top-K → AWN → demod → CRC")
    print(f"{'='*75}")

    # Oracle demod
    print(f"\n  Oracle Demod (correct mod type):")
    oracle_sc = ['Clean', 'CW'] + [f'CW+Top{k}' for k in topk_values]
    header = f"  {'Mod':<6}"
    for sc in oracle_sc:
        header += f" {sc:>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for mod in digital:
        line = f"  {mod:<6}"
        for sc in oracle_sc:
            r = [x for x in rows if x['mod'] == mod and x['scenario'] == sc]
            val = r[0]['crc_pass_rate'] * 100 if r else 0
            line += f" {val:>9.1f}%"
        print(line)

    # AMC demod
    print(f"\n  AMC Demod (AWN-driven):")
    amc_sc = ['Clean+AMC', 'CW+AMC'] + [f'CW+Top{k}+AMC' for k in topk_values]
    header = f"  {'Mod':<6}"
    for sc in amc_sc:
        header += f" {sc:>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for mod in digital:
        line = f"  {mod:<6}"
        for sc in amc_sc:
            r = [x for x in rows if x['mod'] == mod and x['scenario'] == sc]
            val = r[0]['crc_pass_rate'] * 100 if r else 0
            line += f" {val:>13.1f}%"
        print(line)

    # AMC accuracy
    print(f"\n  AWN AMC Accuracy:")
    acc_sc = ['Clean', 'CW'] + [f'CW+Top{k}+AMC' for k in topk_values]
    header = f"  {'Mod':<6}"
    for sc in acc_sc:
        header += f" {sc:>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for mod in digital:
        line = f"  {mod:<6}"
        for sc in acc_sc:
            r = [x for x in rows if x['mod'] == mod and x['scenario'] == sc]
            if r and r[0]['amc_acc'] >= 0:
                val = r[0]['amc_acc'] * 100
                line += f" {val:>13.1f}%"
            else:
                line += f" {'---':>14}"
        print(line)

    # Defense cost on clean
    print(f"\n  Defense Cost (Top-K on clean signal, Oracle demod):")
    clean_sc = ['Clean'] + [f'Clean+Top{k}' for k in topk_values]
    header = f"  {'Mod':<6}"
    for sc in clean_sc:
        header += f" {sc:>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for mod in digital:
        line = f"  {mod:<6}"
        for sc in clean_sc:
            r = [x for x in rows if x['mod'] == mod and x['scenario'] == sc]
            val = r[0]['crc_pass_rate'] * 100 if r else 0
            line += f" {val:>11.1f}%"
        print(line)


def plot_results(rows, mods, topk_values, snr, output_dir):
    """Plot CRC pass rate across scenarios."""
    digital = [m for m in mods if m in ALL_DIGITAL_MODS]
    colors = plt.cm.tab10(np.linspace(0, 1, len(digital)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Oracle demod — Clean vs CW vs CW+TopK
    ax = axes[0]
    for mi, mod in enumerate(digital):
        clean = [x for x in rows if x['mod'] == mod and x['scenario'] == 'Clean']
        cw = [x for x in rows if x['mod'] == mod and x['scenario'] == 'CW']
        clean_crc = clean[0]['crc_pass_rate'] * 100 if clean else 0
        cw_crc = cw[0]['crc_pass_rate'] * 100 if cw else 0

        pts = [(0, cw_crc)]
        for k in topk_values:
            r = [x for x in rows if x['mod'] == mod
                 and x['scenario'] == f'CW+Top{k}']
            pts.append((k, r[0]['crc_pass_rate'] * 100 if r else 0))

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, 'o-', color=colors[mi], label=mod, linewidth=2)
        ax.axhline(y=clean_crc, color=colors[mi], linestyle='--', alpha=0.3)

    ax.set_xlabel('Top-K (0 = no defense)')
    ax.set_ylabel('CRC Pass Rate (%)')
    ax.set_title('Oracle Demod: CW + FFT Recovery')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: AMC demod — CW+AMC vs CW+TopK+AMC
    ax = axes[1]
    for mi, mod in enumerate(digital):
        cw_amc = [x for x in rows if x['mod'] == mod
                  and x['scenario'] == 'CW+AMC']
        cw_amc_crc = cw_amc[0]['crc_pass_rate'] * 100 if cw_amc else 0

        pts = [(0, cw_amc_crc)]
        for k in topk_values:
            r = [x for x in rows if x['mod'] == mod
                 and x['scenario'] == f'CW+Top{k}+AMC']
            pts.append((k, r[0]['crc_pass_rate'] * 100 if r else 0))

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, 'o-', color=colors[mi], label=mod, linewidth=2)

    ax.set_xlabel('Top-K (0 = no defense)')
    ax.set_ylabel('CRC Pass Rate (%)')
    ax.set_title('AMC Demod: CW + FFT Recovery')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Defense cost — Clean vs Clean+TopK
    ax = axes[2]
    for mi, mod in enumerate(digital):
        clean = [x for x in rows if x['mod'] == mod and x['scenario'] == 'Clean']
        clean_crc = clean[0]['crc_pass_rate'] * 100 if clean else 0

        pts = [(0, clean_crc)]
        for k in topk_values:
            r = [x for x in rows if x['mod'] == mod
                 and x['scenario'] == f'Clean+Top{k}']
            pts.append((k, r[0]['crc_pass_rate'] * 100 if r else 0))

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, 'o-', color=colors[mi], label=mod, linewidth=2)

    ax.set_xlabel('Top-K (0 = no defense)')
    ax.set_ylabel('CRC Pass Rate (%)')
    ax.set_title('Defense Cost: Top-K on Clean Signal')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Direct End-to-End CRC Defense Pipeline (SNR={snr} dB)',
                 fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'fig_crc_defense_pipeline.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Direct End-to-End CRC Defense Pipeline')
    parser.add_argument('--snr', type=int, default=18)
    parser.add_argument('--n_bursts', type=int, default=200)
    parser.add_argument('--topk', type=str, default='10,20,50',
                        help='Comma-separated Top-K values')
    parser.add_argument('--mods', type=str,
                        default='BPSK,QPSK,8PSK,QAM16,QAM64,PAM4')
    parser.add_argument('--target_rms', type=float, default=DEFAULT_TARGET_RMS,
                        help='Target RMS for synthetic signals (default: 0.0082)')
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_steps', type=int, default=1000)
    parser.add_argument('--ckpt', type=str, default='./checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default='./results/crc_defense_direct')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run(args)
