#!/usr/bin/env python
"""
Multi-Attack FEC CRC Defense Pipeline:
    synth(CRC+FEC) -> IQ -> attack -> FFT Top-K -> demod(FEC) -> CRC

Tests multiple Linf attacks (not just CW) on the FEC CRC pipeline.
Reference: results/crc_defense_fec/FEC_EXPERIMENT_REPORT.md

Attacks tested: apgd, bim, mifgsm, pgd, rfgsm, vmifgsm, vnifgsm
(All Linf-bounded iterative attacks from torchattacks)

Usage:
    python crc_defense_fec_multi_attack.py
    python crc_defense_fec_multi_attack.py --attacks pgd,bim --snr 18
    python crc_defense_fec_multi_attack.py --n_bursts 100 --snr 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from util.synth_txrx import (
    generate_burst, demodulate_burst, get_bits_per_symbol,
    ALL_DIGITAL_MODS, fec_payload_capacity,
)
from util.defense import fft_topk_denoise
from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial

IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}

DEFAULT_TARGET_RMS = 0.0082


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
        adv_list.append(x_adv.detach().cpu())
    return torch.cat(adv_list, dim=0).numpy()


def apply_fft_topk_batch(iq_batch_np, topk):
    x_t = torch.from_numpy(iq_batch_np).float()
    x_filt = fft_topk_denoise(x_t, topk=topk)
    return x_filt.numpy()


def tensor_to_complex(iq_tensor_2d):
    return iq_tensor_2d[0, :] + 1j * iq_tensor_2d[1, :]


def demod_burst(burst_info, use_mod_type, override_iq_complex=None,
                fec=False, soft_demod=True):
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


def run_attack_scenarios(mod, snr, fec_flag, n_bursts, topk_values,
                         target_rms, model, wrapped_model, attack,
                         attack_name, device, seed):
    """Run scenarios for one (mod, snr, fec, attack) combination.

    Scenarios:
      1. Clean + Oracle
      2. Attack + Oracle  (attack doesn't corrupt data plane?)
      3. Attack + Top-K + Oracle  (defense recovers CRC?)
    """
    true_idx = MOD_TO_IDX[mod]
    rng = np.random.default_rng(seed + hash((mod, snr)) % 2**31)

    bursts = []
    for _ in range(n_bursts):
        b = generate_burst(mod, n_symbols=16, n_pilots=2, sps=8,
                           beta=0.35, snr_db=snr, target_rms=target_rms,
                           cfo_std=0.0, rng=rng, n_guard=16, fec=fec_flag)
        bursts.append(b)

    actual_fec = bursts[0]['fec']
    iq_batch = np.concatenate([b['iq_tensor'] for b in bursts], axis=0)

    # Clean AMC accuracy
    clean_preds = classify_batch(model, iq_batch, device)
    clean_acc = float(np.mean(clean_preds == true_idx))

    # Attack
    labels = np.full(n_bursts, true_idx)
    adv_batch = attack_batch(attack, iq_batch, labels,
                             wrapped_model, device, ta_box='minmax')
    adv_preds = classify_batch(model, adv_batch, device)
    adv_acc = float(np.mean(adv_preds == true_idx))

    results = {}

    # 1. Clean + Oracle
    crc_cnt = sum(
        int(demod_burst(bursts[i], mod, fec=actual_fec)['crc_pass'])
        for i in range(n_bursts))
    results['Clean'] = crc_cnt / n_bursts

    # 2. Attack + Oracle (control-plane check)
    crc_cnt = 0
    for i in range(n_bursts):
        adv_iq = tensor_to_complex(adv_batch[i])
        r = demod_burst(bursts[i], mod, override_iq_complex=adv_iq,
                        fec=actual_fec)
        crc_cnt += int(r['crc_pass'])
    results['Att+Oracle'] = crc_cnt / n_bursts

    # 3. Attack + Top-K + Oracle
    topk_accs = {}
    for k in topk_values:
        rec = apply_fft_topk_batch(adv_batch, topk=k)
        rec_p = classify_batch(model, rec, device)
        rec_acc = float(np.mean(rec_p == true_idx))
        topk_accs[k] = rec_acc

        crc_cnt = 0
        for i in range(n_bursts):
            rec_iq = tensor_to_complex(rec[i])
            r = demod_burst(bursts[i], mod, override_iq_complex=rec_iq,
                            fec=actual_fec)
            crc_cnt += int(r['crc_pass'])
        results['Top%d' % k] = crc_cnt / n_bursts

    return results, clean_acc, adv_acc, topk_accs, actual_fec


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Attack FEC CRC Defense Pipeline')
    parser.add_argument('--snr', type=str, default='18,0')
    parser.add_argument('--n_bursts', type=int, default=200)
    parser.add_argument('--topk', type=str, default='10,20,50')
    parser.add_argument('--mods', type=str,
                        default='BPSK,QPSK,8PSK,QAM16,QAM64,PAM4')
    parser.add_argument('--attacks', type=str,
                        default='apgd,bim,mifgsm,pgd,rfgsm,vmifgsm,vnifgsm')
    parser.add_argument('--attack_eps', type=float, default=0.1)
    parser.add_argument('--target_rms', type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument('--ckpt', type=str, default='./checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default='./results/crc_defense_fec')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

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

    # Attack config
    cfg.attack_eps = args.attack_eps
    cfg.ta_box = 'minmax'
    cfg.num_classes = 11
    # CW-specific (in case user includes cw)
    cfg.cw_c = 10.0
    cfg.cw_steps = 200
    cfg.cw_lr = 0.005

    snr_list = [int(x) for x in args.snr.split(',')]
    topk_values = [int(x) for x in args.topk.split(',')]
    mods = args.mods.split(',')
    attack_names = [a.strip() for a in args.attacks.split(',')]

    print('=' * 90)
    print('  Multi-Attack FEC CRC Defense Pipeline')
    print('=' * 90)
    print('  Attacks:     %s' % attack_names)
    print('  Eps:         %.3f (minmax)' % args.attack_eps)
    print('  SNRs:        %s dB' % snr_list)
    print('  Top-K:       %s' % topk_values)
    print('  Mods:        %s' % mods)
    print('  N bursts:    %d' % args.n_bursts)
    print('  Device:      %s' % device)
    print()

    # Verify attacks
    attacks = {}
    for name in attack_names:
        try:
            atk = create_attack(name, wrapped_model, cfg)
            attacks[name] = atk
            print('  [OK] %s -> %s' % (name, type(atk).__name__))
        except Exception as e:
            print('  [FAIL] %s -> %s' % (name, e))

    if not attacks:
        print('No valid attacks. Exiting.')
        return

    print()

    # Collect all results
    all_results = []

    for snr in snr_list:
        for attack_name, attack in attacks.items():
            print('\n' + '=' * 90)
            print('  %s | SNR=%d dB' % (attack_name.upper(), snr))
            print('=' * 90)

            # Header
            topk_cols = ['Top%d' % k for k in topk_values]
            hdr = '%-6s %-5s %8s %8s %10s' % (
                'Mod', 'FEC', 'ClnAMC', 'AttAMC', 'Att+Oracl')
            for tc in topk_cols:
                hdr += ' %8s' % tc
            print(hdr)
            print('-' * (50 + 9 * len(topk_values)))

            for mod in mods:
                if mod not in ALL_DIGITAL_MODS:
                    continue

                for fec_flag in [False, True]:
                    bps = get_bits_per_symbol(mod)
                    _, can_fec = fec_payload_capacity(14, bps)
                    if fec_flag and not can_fec:
                        continue

                    fec_label = 'FEC' if fec_flag else 'noFEC'

                    t0 = time.time()
                    res, clean_acc, adv_acc, topk_accs, actual_fec = \
                        run_attack_scenarios(
                            mod, snr, fec_flag, args.n_bursts, topk_values,
                            args.target_rms, model, wrapped_model, attack,
                            attack_name, device, args.seed)
                    elapsed = time.time() - t0

                    row = '%-6s %-5s %7.1f%% %7.1f%% %9.1f%%' % (
                        mod, fec_label,
                        clean_acc * 100, adv_acc * 100,
                        res['Att+Oracle'] * 100)
                    for k in topk_values:
                        row += ' %7.1f%%' % (res['Top%d' % k] * 100)
                    row += '  (%.1fs)' % elapsed
                    print(row)

                    # Store
                    entry = {
                        'attack': attack_name,
                        'mod': mod,
                        'snr': snr,
                        'fec': actual_fec,
                        'clean_amc_acc': round(clean_acc, 4),
                        'attack_amc_acc': round(adv_acc, 4),
                        'attack_oracle_crc': round(res['Att+Oracle'], 4),
                        'clean_crc': round(res['Clean'], 4),
                    }
                    for k in topk_values:
                        entry['top%d_crc' % k] = round(res['Top%d' % k], 4)
                        entry['top%d_amc' % k] = round(topk_accs[k], 4)
                    all_results.append(entry)

    # Save results
    csv_path = os.path.join(args.output_dir, 'crc_multi_attack_fec.csv')
    if all_results:
        cols = list(all_results[0].keys())
        with open(csv_path, 'w') as f:
            f.write(','.join(cols) + '\n')
            for r in all_results:
                f.write(','.join(str(r[c]) for c in cols) + '\n')
        print('\nSaved: %s' % csv_path)

    json_path = os.path.join(args.output_dir, 'crc_multi_attack_fec.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print('Saved: %s' % json_path)

    # Print summary table
    print_summary(all_results, attack_names, mods, snr_list, topk_values)


def print_summary(all_results, attack_names, mods, snr_list, topk_values):
    """Print summary comparison table."""
    print('\n' + '=' * 90)
    print('  SUMMARY: Attack+Oracle CRC (control-plane check)')
    print('  If CRC ~= Clean → attack is control-plane only')
    print('=' * 90)

    for snr in snr_list:
        print('\n  SNR = %d dB' % snr)
        hdr = '  %-6s %-5s %8s' % ('Mod', 'FEC', 'Clean')
        for atk in attack_names:
            hdr += ' %8s' % atk[:8]
        print(hdr)
        print('  ' + '-' * (22 + 9 * len(attack_names)))

        for mod in mods:
            if mod not in ALL_DIGITAL_MODS:
                continue
            for fec_val in [False, True]:
                fec_label = 'FEC' if fec_val else 'noFEC'
                clean_rows = [r for r in all_results
                              if r['mod'] == mod and r['snr'] == snr
                              and r['fec'] == fec_val]
                if not clean_rows:
                    continue
                clean_crc = clean_rows[0]['clean_crc'] * 100

                line = '  %-6s %-5s %7.1f%%' % (mod, fec_label, clean_crc)
                for atk in attack_names:
                    rows = [r for r in all_results
                            if r['attack'] == atk and r['mod'] == mod
                            and r['snr'] == snr and r['fec'] == fec_val]
                    if rows:
                        line += ' %7.1f%%' % (rows[0]['attack_oracle_crc'] * 100)
                    else:
                        line += ' %8s' % '---'
                print(line)

    print('\n' + '=' * 90)
    print('  SUMMARY: Top-K Recovery CRC (FEC)')
    print('=' * 90)

    for snr in snr_list:
        for k in topk_values:
            print('\n  SNR=%d, Top-%d, FEC=True:' % (snr, k))
            hdr = '  %-6s' % 'Mod'
            for atk in attack_names:
                hdr += ' %8s' % atk[:8]
            print(hdr)
            print('  ' + '-' * (8 + 9 * len(attack_names)))

            for mod in mods:
                if mod not in ALL_DIGITAL_MODS:
                    continue
                line = '  %-6s' % mod
                for atk in attack_names:
                    rows = [r for r in all_results
                            if r['attack'] == atk and r['mod'] == mod
                            and r['snr'] == snr and r['fec'] == True]
                    if rows:
                        key = 'top%d_crc' % k
                        if key in rows[0]:
                            line += ' %7.1f%%' % (rows[0][key] * 100)
                        else:
                            line += ' %8s' % '---'
                    else:
                        line += ' %8s' % '---'
                print(line)


if __name__ == '__main__':
    main()
