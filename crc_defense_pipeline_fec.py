#!/usr/bin/env python
"""
FEC-Enabled CRC Defense Pipeline:
    synth(CRC+FEC) -> IQ -> AWN -> CW attack -> FFT Top-K -> AWN -> demod(FEC) -> CRC

Compares FEC vs no-FEC side by side across multiple SNRs.

Scenarios (same 7 as crc_defense_pipeline.py):
  1. Clean + Oracle:      synth IQ -> demod(true mod) -> CRC
  2. Clean + AMC:         synth IQ -> AWN -> demod(AWN pred) -> CRC
  3. CW + Oracle:         synth IQ -> CW -> demod(true mod) -> CRC
  4. CW + AMC:            synth IQ -> CW -> AWN -> demod(AWN pred) -> CRC
  5. CW + Top-K + Oracle: synth IQ -> CW -> Top-K -> demod(true mod) -> CRC
  6. CW + Top-K + AMC:    synth IQ -> CW -> Top-K -> AWN -> demod(AWN pred) -> CRC
  7. Clean + Top-K:       synth IQ -> Top-K -> demod(true mod) -> CRC  (defense cost)

Each scenario is run with FEC enabled and disabled for comparison.

Usage:
    python crc_defense_pipeline_fec.py --snr 18,0
    python crc_defense_pipeline_fec.py --snr 18,0 --topk 10,20,50
    python crc_defense_pipeline_fec.py --snr 0 --n_bursts 500 --mods QPSK,8PSK,QAM16
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
    fec_payload_capacity,
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

DEFAULT_TARGET_RMS = 0.0082


# ============================================================
# Helpers (same as crc_defense_pipeline.py)
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


def apply_fft_topk_batch(iq_batch_np, topk):
    x_t = torch.from_numpy(iq_batch_np).float()
    x_filt = fft_topk_denoise(x_t, topk=topk)
    return x_filt.numpy()


def tensor_to_complex(iq_tensor_2d):
    return iq_tensor_2d[0, :] + 1j * iq_tensor_2d[1, :]


def demod_burst(burst_info, use_mod_type, override_iq_complex=None,
                fec=False, soft_demod=True):
    """Demodulate burst, optionally with FEC decoding."""
    try:
        iq_full = burst_info.get('iq_full')
        iq_win_start = burst_info.get('iq_win_start')
        n_guard = burst_info.get('n_guard')

        if override_iq_complex is not None and iq_full is not None:
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
            fec=fec, soft_demod=soft_demod,
            fec_coded_len=burst_info.get('fec_coded_len'),
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
# Run scenarios for one (mod, snr, fec) combination
# ============================================================

def run_scenarios(mod, snr, fec_flag, n_bursts, topk_values, target_rms,
                  model, wrapped_model, cw_attack, device, seed):
    """Run all 7 scenarios for a given (mod, snr, fec) combination.

    Returns list of result dicts.
    """
    true_idx = MOD_TO_IDX[mod]
    rng = np.random.default_rng(seed + hash((mod, snr)) % 2**31)

    # Step 1: Generate synthetic bursts
    bursts = []
    for _ in range(n_bursts):
        b = generate_burst(mod, n_symbols=16, n_pilots=2, sps=8,
                           beta=0.35, snr_db=snr, target_rms=target_rms,
                           cfo_std=0.0, rng=rng, n_guard=16, fec=fec_flag)
        bursts.append(b)

    actual_fec = bursts[0]['fec']
    iq_batch = np.concatenate([b['iq_tensor'] for b in bursts], axis=0)

    # Step 2: AWN on clean
    clean_preds = classify_batch(model, iq_batch, device)
    clean_acc = float(np.mean(clean_preds == true_idx))

    # Step 3: CW attack
    labels = np.full(n_bursts, true_idx)
    adv_batch = attack_batch(cw_attack, iq_batch, labels,
                             wrapped_model, device, ta_box='minmax')
    adv_preds = classify_batch(model, adv_batch, device)
    adv_acc = float(np.mean(adv_preds == true_idx))

    # Step 4: FFT Top-K recovery
    topk_recovered = {}
    topk_preds = {}
    topk_accs = {}
    for k in topk_values:
        rec = apply_fft_topk_batch(adv_batch, topk=k)
        rec_p = classify_batch(model, rec, device)
        topk_recovered[k] = rec
        topk_preds[k] = rec_p
        topk_accs[k] = float(np.mean(rec_p == true_idx))

    # Step 5: Demodulate all scenarios
    scenario_results = {}
    fec_label = 'FEC' if actual_fec else 'noFEC'

    # 5a. Clean + Oracle
    crc_cnt = sum(
        int(demod_burst(bursts[i], mod, fec=actual_fec)['crc_pass'])
        for i in range(n_bursts))
    scenario_results['Clean'] = {
        'crc': crc_cnt / n_bursts, 'amc_acc': clean_acc}

    # 5b. Clean + AMC
    crc_cnt = 0
    for i in range(n_bursts):
        cp = int(clean_preds[i])
        cp_mod = IDX_TO_MOD.get(cp)
        if cp_mod and cp_mod in ALL_DIGITAL_MODS:
            r = demod_burst(bursts[i], cp_mod, fec=actual_fec)
        else:
            r = {'crc_pass': False}
        crc_cnt += int(r['crc_pass'])
    scenario_results['Clean+AMC'] = {
        'crc': crc_cnt / n_bursts, 'amc_acc': clean_acc}

    # 5c. CW + Oracle
    crc_cnt = 0
    for i in range(n_bursts):
        adv_iq = tensor_to_complex(adv_batch[i])
        r = demod_burst(bursts[i], mod, override_iq_complex=adv_iq,
                        fec=actual_fec)
        crc_cnt += int(r['crc_pass'])
    scenario_results['CW'] = {
        'crc': crc_cnt / n_bursts, 'amc_acc': adv_acc}

    # 5d. CW + AMC
    crc_cnt = 0
    for i in range(n_bursts):
        ap = int(adv_preds[i])
        ap_mod = IDX_TO_MOD.get(ap)
        adv_iq = tensor_to_complex(adv_batch[i])
        if ap_mod and ap_mod in ALL_DIGITAL_MODS:
            r = demod_burst(bursts[i], ap_mod, override_iq_complex=adv_iq,
                            fec=actual_fec)
        else:
            r = {'crc_pass': False}
        crc_cnt += int(r['crc_pass'])
    scenario_results['CW+AMC'] = {
        'crc': crc_cnt / n_bursts, 'amc_acc': adv_acc}

    # 5e. For each Top-K value
    for k in topk_values:
        rec_batch = topk_recovered[k]
        rec_p = topk_preds[k]
        rec_acc = topk_accs[k]

        # CW + Top-K + Oracle
        crc_cnt = 0
        for i in range(n_bursts):
            rec_iq = tensor_to_complex(rec_batch[i])
            r = demod_burst(bursts[i], mod, override_iq_complex=rec_iq,
                            fec=actual_fec)
            crc_cnt += int(r['crc_pass'])
        scenario_results['CW+Top%d' % k] = {
            'crc': crc_cnt / n_bursts, 'amc_acc': rec_acc}

        # CW + Top-K + AMC
        crc_cnt = 0
        for i in range(n_bursts):
            rec_iq = tensor_to_complex(rec_batch[i])
            rp = int(rec_p[i])
            rp_mod = IDX_TO_MOD.get(rp)
            if rp_mod and rp_mod in ALL_DIGITAL_MODS:
                r = demod_burst(bursts[i], rp_mod,
                                override_iq_complex=rec_iq, fec=actual_fec)
            else:
                r = {'crc_pass': False}
            crc_cnt += int(r['crc_pass'])
        scenario_results['CW+Top%d+AMC' % k] = {
            'crc': crc_cnt / n_bursts, 'amc_acc': rec_acc}

        # Clean + Top-K + Oracle (defense cost)
        clean_topk = apply_fft_topk_batch(iq_batch, topk=k)
        crc_cnt = 0
        for i in range(n_bursts):
            filt_iq = tensor_to_complex(clean_topk[i])
            r = demod_burst(bursts[i], mod, override_iq_complex=filt_iq,
                            fec=actual_fec)
            crc_cnt += int(r['crc_pass'])
        scenario_results['Clean+Top%d' % k] = {
            'crc': crc_cnt / n_bursts}

    # Convert to row dicts
    rows = []
    for sc_name, sc_data in scenario_results.items():
        rows.append({
            'mod': mod, 'snr': snr, 'fec': actual_fec,
            'scenario': sc_name,
            'crc_pass_rate': round(sc_data['crc'], 4),
            'amc_acc': round(sc_data.get('amc_acc', -1), 4),
        })
    return rows, clean_acc, adv_acc, topk_accs


# ============================================================
# Printing & plotting
# ============================================================

def print_comparison_table(all_rows, mods, topk_values, snr_list):
    """Print FEC vs no-FEC comparison table."""
    digital = [m for m in mods if m in ALL_DIGITAL_MODS]

    for snr in snr_list:
        print('\n' + '=' * 80)
        print('  FEC vs no-FEC Comparison (SNR=%d dB)' % snr)
        print('=' * 80)

        # Oracle demod scenarios
        oracle_sc = ['Clean', 'CW'] + ['CW+Top%d' % k for k in topk_values]
        header = '  %-6s' % 'Mod'
        for sc in oracle_sc:
            header += '  %s' % sc.center(19)
        print('\n  Oracle Demod:')
        print(header)

        sub_header = '  %-6s' % ''
        for sc in oracle_sc:
            sub_header += '  %s  %s' % ('noFEC'.center(9), 'FEC'.center(8))
        print(sub_header)
        print('  ' + '-' * (len(sub_header) - 2))

        for mod in digital:
            line = '  %-6s' % mod
            for sc in oracle_sc:
                for fec_val in [False, True]:
                    r = [x for x in all_rows
                         if x['mod'] == mod and x['snr'] == snr
                         and x['scenario'] == sc and x['fec'] == fec_val]
                    if r:
                        val = r[0]['crc_pass_rate'] * 100
                        line += ' %8.1f%%' % val
                    else:
                        line += ' %9s' % '---'
            print(line)

        # AMC demod scenarios
        amc_sc = ['Clean+AMC', 'CW+AMC'] + \
            ['CW+Top%d+AMC' % k for k in topk_values]
        print('\n  AMC Demod:')
        header = '  %-6s' % 'Mod'
        for sc in amc_sc:
            header += '  %s' % sc.center(19)
        print(header)

        sub_header = '  %-6s' % ''
        for sc in amc_sc:
            sub_header += '  %s  %s' % ('noFEC'.center(9), 'FEC'.center(8))
        print(sub_header)
        print('  ' + '-' * (len(sub_header) - 2))

        for mod in digital:
            line = '  %-6s' % mod
            for sc in amc_sc:
                for fec_val in [False, True]:
                    r = [x for x in all_rows
                         if x['mod'] == mod and x['snr'] == snr
                         and x['scenario'] == sc and x['fec'] == fec_val]
                    if r:
                        val = r[0]['crc_pass_rate'] * 100
                        line += ' %8.1f%%' % val
                    else:
                        line += ' %9s' % '---'
            print(line)


def plot_fec_comparison(all_rows, mods, topk_values, snr_list, output_dir):
    """Plot FEC vs no-FEC comparison."""
    digital = [m for m in mods if m in ALL_DIGITAL_MODS]
    n_snrs = len(snr_list)

    fig, axes = plt.subplots(n_snrs, 3, figsize=(18, 6 * n_snrs),
                             squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(digital)))

    for si, snr in enumerate(snr_list):
        # Panel 1: Oracle CRC — noFEC vs FEC (CW + TopK recovery)
        ax = axes[si, 0]
        for mi, mod in enumerate(digital):
            for fec_val, ls in [(False, '--'), (True, '-')]:
                pts = []
                cw = [x for x in all_rows
                      if x['mod'] == mod and x['snr'] == snr
                      and x['scenario'] == 'CW' and x['fec'] == fec_val]
                if cw:
                    pts.append((0, cw[0]['crc_pass_rate'] * 100))
                for k in topk_values:
                    r = [x for x in all_rows
                         if x['mod'] == mod and x['snr'] == snr
                         and x['scenario'] == 'CW+Top%d' % k
                         and x['fec'] == fec_val]
                    if r:
                        pts.append((k, r[0]['crc_pass_rate'] * 100))
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    label = '%s %s' % (mod, 'FEC' if fec_val else 'noFEC')
                    ax.plot(xs, ys, 'o' + ls, color=colors[mi],
                            label=label, linewidth=2,
                            alpha=1.0 if fec_val else 0.5)
        ax.set_xlabel('Top-K (0 = no defense)')
        ax.set_ylabel('CRC Pass Rate (%)')
        ax.set_title('Oracle: CW Recovery (SNR=%d)' % snr)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

        # Panel 2: Clean Oracle — FEC vs noFEC
        ax = axes[si, 1]
        bar_width = 0.35
        x_pos = np.arange(len(digital))
        for fi, (fec_val, label, offset) in enumerate(
                [(False, 'noFEC', -bar_width/2),
                 (True, 'FEC', bar_width/2)]):
            vals = []
            for mod in digital:
                r = [x for x in all_rows
                     if x['mod'] == mod and x['snr'] == snr
                     and x['scenario'] == 'Clean' and x['fec'] == fec_val]
                vals.append(r[0]['crc_pass_rate'] * 100 if r else 0)
            ax.bar(x_pos + offset, vals, bar_width, label=label,
                   alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(digital, rotation=45, ha='right')
        ax.set_ylabel('CRC Pass Rate (%)')
        ax.set_title('Clean Oracle (SNR=%d)' % snr)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 3: FEC improvement (delta) for CW+TopK Oracle
        ax = axes[si, 2]
        for mi, mod in enumerate(digital):
            deltas = []
            ks = []
            for k in topk_values:
                nofec = [x for x in all_rows
                         if x['mod'] == mod and x['snr'] == snr
                         and x['scenario'] == 'CW+Top%d' % k
                         and x['fec'] == False]
                fec = [x for x in all_rows
                       if x['mod'] == mod and x['snr'] == snr
                       and x['scenario'] == 'CW+Top%d' % k
                       and x['fec'] == True]
                if nofec and fec:
                    delta = (fec[0]['crc_pass_rate'] -
                             nofec[0]['crc_pass_rate']) * 100
                    deltas.append(delta)
                    ks.append(k)
            if deltas:
                ax.bar([str(k) for k in ks], deltas, color=colors[mi],
                       alpha=0.7, label=mod)
        ax.set_xlabel('Top-K')
        ax.set_ylabel('CRC Improvement (FEC - noFEC) pp')
        ax.set_title('FEC Coding Gain (SNR=%d)' % snr)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linewidth=0.5)

    plt.suptitle('FEC vs no-FEC CRC Defense Comparison', fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'fig_fec_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: %s' % fig_path)


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

    snr_list = [int(x) for x in args.snr.split(',')]
    topk_values = [int(x) for x in args.topk.split(',')]
    mods = args.mods.split(',')
    target_rms = args.target_rms
    n_bursts = args.n_bursts

    print('=' * 80)
    print('  FEC-Enabled CRC Defense Pipeline')
    print('  synth(CRC+FEC) -> IQ -> AWN -> CW -> Top-K -> AWN -> demod(FEC) -> CRC')
    print('=' * 80)
    print('  SNRs:        %s dB' % snr_list)
    print('  Top-K:       %s' % topk_values)
    print('  Mods:        %s' % mods)
    print('  N bursts:    %d' % n_bursts)
    print('  Target RMS:  %s' % target_rms)
    print('  CW:          c=%s, steps=%s' % (args.cw_c, args.cw_steps))
    print('  Device:      %s' % device)

    all_rows = []

    for snr in snr_list:
        for mod in mods:
            if mod not in ALL_DIGITAL_MODS:
                print('\n  Skipping %s (not a digital mod)' % mod)
                continue

            for fec_flag in [False, True]:
                fec_label = 'FEC' if fec_flag else 'noFEC'

                # Check if FEC is even possible for this mod
                bps = get_bits_per_symbol(mod)
                _, can_fec = fec_payload_capacity(14, bps)
                if fec_flag and not can_fec:
                    print('\n  %s SNR=%d %s: skipped (FEC not possible, '
                          '1 bps)' % (mod, snr, fec_label))
                    continue

                print('\n%s' % ('=' * 80))
                print('  %s (SNR=%d dB, %s)' % (mod, snr, fec_label))
                print('%s' % ('=' * 80))

                t0 = time.time()
                rows, clean_acc, adv_acc, topk_accs = run_scenarios(
                    mod, snr, fec_flag, n_bursts, topk_values, target_rms,
                    model, wrapped_model, cw_attack, device, args.seed)
                elapsed = time.time() - t0

                print('  Clean AMC: %.1f%%, CW AMC: %.1f%% (%.1fs)' %
                      (100 * clean_acc, 100 * adv_acc, elapsed))

                # Print scenario CRC rates
                for r in rows:
                    sc = r['scenario']
                    crc = r['crc_pass_rate'] * 100
                    print('    %-22s CRC: %6.1f%%' % (sc, crc))

                all_rows.extend(rows)

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'crc_defense_fec.csv')
    cols = ['mod', 'snr', 'fec', 'scenario', 'crc_pass_rate', 'amc_acc']
    with open(csv_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in all_rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print('\n  Saved: %s' % csv_path)

    # Save JSON
    json_path = os.path.join(args.output_dir, 'crc_defense_fec.json')
    with open(json_path, 'w') as f:
        json.dump(all_rows, f, indent=2)
    print('  Saved: %s' % json_path)

    # Print comparison table
    print_comparison_table(all_rows, mods, topk_values, snr_list)

    # Plot
    plot_fec_comparison(all_rows, mods, topk_values, snr_list, args.output_dir)

    # Summary report
    print_summary_report(all_rows, mods, topk_values, snr_list)

    return all_rows


def print_summary_report(all_rows, mods, topk_values, snr_list):
    """Print a concise summary of FEC coding gain."""
    digital = [m for m in mods if m in ALL_DIGITAL_MODS]

    print('\n' + '=' * 80)
    print('  FEC CODING GAIN SUMMARY')
    print('=' * 80)

    for snr in snr_list:
        print('\n  SNR = %d dB:' % snr)
        print('  %-6s  %-12s  %8s  %8s  %8s' %
              ('Mod', 'Scenario', 'noFEC', 'FEC', 'Gain'))
        print('  ' + '-' * 50)

        for mod in digital:
            for sc in ['Clean'] + ['CW+Top%d' % k for k in topk_values]:
                nofec = [x for x in all_rows
                         if x['mod'] == mod and x['snr'] == snr
                         and x['scenario'] == sc and x['fec'] == False]
                fec = [x for x in all_rows
                       if x['mod'] == mod and x['snr'] == snr
                       and x['scenario'] == sc and x['fec'] == True]
                if nofec and fec:
                    nv = nofec[0]['crc_pass_rate'] * 100
                    fv = fec[0]['crc_pass_rate'] * 100
                    gain = fv - nv
                    sign = '+' if gain >= 0 else ''
                    print('  %-6s  %-12s  %7.1f%%  %7.1f%%  %s%.1f pp' %
                          (mod, sc, nv, fv, sign, gain))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FEC-Enabled CRC Defense Pipeline')
    parser.add_argument('--snr', type=str, default='18,0',
                        help='Comma-separated SNR values (dB)')
    parser.add_argument('--n_bursts', type=int, default=200)
    parser.add_argument('--topk', type=str, default='10,20,50',
                        help='Comma-separated Top-K values')
    parser.add_argument('--mods', type=str,
                        default='BPSK,QPSK,8PSK,QAM16,QAM64,PAM4')
    parser.add_argument('--target_rms', type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument('--cw_c', type=float, default=1.0)
    parser.add_argument('--cw_steps', type=int, default=1000)
    parser.add_argument('--ckpt', type=str, default='./checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default='./results/crc_defense_fec')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run(args)
