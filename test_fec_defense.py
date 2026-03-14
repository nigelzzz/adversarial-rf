#!/usr/bin/env python
"""
Test whether FEC improves data recovery after defense filtering.

Pipeline:
  TX: data -> [FEC encode] -> modulate -> pulse shape -> channel(SNR)
  Attack: adversarial perturbation on IQ signal
  Defense: apply spectral/time-domain filter to remove perturbation
  RX: matched filter -> demod -> [FEC decode] -> CRC check

Compare CRC pass rate: no-FEC vs FEC, across defenses.
"""
import argparse
import numpy as np
import torch
import pickle
from collections import Counter

from util.synth_txrx import (
    generate_burst, demodulate_burst, get_bits_per_symbol,
    CONSTELLATION_MODS, FSK_MODS,
)
from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial, compute_accuracy
from util.defense import fft_topk_denoise

# Defense functions (from test_alt_defenses.py)
import torch.nn.functional as F

def moving_avg_smooth(x, kernel_size=5):
    N, C, T = x.shape
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
    x_flat = x.view(N * C, 1, T)
    x_smooth = F.conv1d(x_flat, kernel, padding=pad)
    return x_smooth.view(N, C, T)

def input_quantize(x, n_levels=32):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    x_quant = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
    return x_quant * (x_max - x_min) + x_min

def median_filter(x, kernel_size=5):
    N, C, T = x.shape
    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    windows = x_padded.unfold(dimension=2, size=kernel_size, step=1)
    return windows.median(dim=3).values

IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}


def apply_defense_to_iq(defense_name, iq_tensor_torch, device):
    """Apply defense to [N, 2, T] torch tensor, return torch tensor."""
    x = iq_tensor_torch.to(device)
    if defense_name == 'none':
        return x
    elif defense_name == 'top10':
        return fft_topk_denoise(x, topk=10)
    elif defense_name == 'top20':
        return fft_topk_denoise(x, topk=20)
    elif defense_name == 'quant32':
        return input_quantize(x, n_levels=32)
    elif defense_name == 'mvavg5':
        return moving_avg_smooth(x, kernel_size=5)
    elif defense_name == 'median5':
        return median_filter(x, kernel_size=5)
    else:
        raise ValueError(f"Unknown defense: {defense_name}")


def demod_burst_with_defense(burst_info, adv_iq_complex, mod_type, fec):
    """Demodulate a single burst after defense filtering."""
    try:
        iq_full = burst_info.get('iq_full')
        iq_win_start = burst_info.get('iq_win_start')
        n_guard = burst_info.get('n_guard')

        # Splice adversarial/defended IQ into full signal
        if iq_full is not None:
            iq_full = iq_full.copy()
            n_win = len(adv_iq_complex)
            iq_full[iq_win_start:iq_win_start + n_win] = adv_iq_complex

        result = demodulate_burst(
            adv_iq_complex,
            mod_type,
            n_pilots=burst_info['n_pilots'],
            pilot_symbols=burst_info.get('pilot_symbols'),
            pilot_bits=burst_info.get('pilot_bits'),
            sps=8, beta=0.35,
            pilot_positions=burst_info.get('pilot_positions'),
            iq_full=iq_full,
            iq_win_start=iq_win_start,
            n_guard=n_guard,
            fec=fec,
            soft_demod=True,
            fec_coded_len=burst_info.get('fec_coded_len'),
        )
        return result['crc_pass']
    except Exception:
        return False


def main():
    fix_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    cfg = Config('2016.10a', train=False)
    cfg.device = device
    model = create_model(cfg, model_name='awn')
    model.load_state_dict(torch.load('./checkpoint/2016.10a_AWN.pkl', map_location=device))
    model.eval()

    cfg.cw_c = 10.0
    cfg.cw_steps = 200
    cfg.cw_lr = 0.005
    cfg.attack_eps = 0.1
    cfg.ta_box = 'minmax'
    cfg.num_classes = 11

    wrapped_model = Model01Wrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # Load real data for perturbation transfer
    with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
        rml_data = pickle.load(f, encoding='bytes')

    # Config
    mods_to_test = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4', 'CPFSK', 'GFSK']
    snrs = [0, 18]
    n_bursts = 200
    attacks = ['cw', 'eadl1']
    defenses = ['none', 'top10', 'top20', 'quant32', 'mvavg5', 'median5']

    # Header
    def_header = "".join(f"{d:>10}" for d in defenses)
    print("=" * 120)
    print("  FEC vs No-FEC: CRC Pass Rate After Defense Filtering")
    print("  (Oracle demod — correct modulation always used)")
    print("=" * 120)

    for attack_name in attacks:
        attack = create_attack(attack_name, wrapped_model, cfg)

        for snr_db in snrs:
            print(f"\n{'='*120}")
            print(f"  Attack: {attack_name.upper()}, SNR: {snr_db} dB")
            print(f"{'='*120}")
            print(f"{'Mod':<8}{'FEC':>5}" + def_header)
            print("-" * 120)

            for mod_type in mods_to_test:
                bps = get_bits_per_symbol(mod_type)

                # Check if FEC is feasible (need enough coded capacity)
                n_data_symbols = 16 - 2  # n_symbols - n_pilots
                coded_capacity = n_data_symbols * bps
                payload_bits = coded_capacity // 2 - 6  # K-1=6 tail bits
                can_fec = (payload_bits >= 8) and (mod_type not in FSK_MODS)

                # Generate real adversarial perturbations
                key = (mod_type.encode(), snr_db)
                if key not in rml_data:
                    continue
                real_samples = rml_data[key][:200]
                real_t = torch.from_numpy(real_samples).float().to(device)
                true_idx = MOD_TO_IDX[mod_type]
                y_real = torch.full((len(real_t),), true_idx, dtype=torch.long, device=device)

                x_adv_real = generate_adversarial(
                    attack, real_t, y_real,
                    wrapped_model=wrapped_model, ta_box='minmax',
                )
                # Compute perturbation delta
                delta_real = (x_adv_real - real_t).cpu().numpy()

                rng = np.random.default_rng(42)

                for fec_mode in [False, True]:
                    if fec_mode and not can_fec:
                        # Print N/A row for FEC when not feasible
                        row = f"{mod_type:<8}{'Y':>5}"
                        for _ in defenses:
                            row += f"{'N/A':>10}"
                        print(row)
                        continue

                    # Generate synthetic bursts
                    bursts = []
                    for _ in range(n_bursts):
                        b = generate_burst(
                            mod_type, n_symbols=16, n_pilots=2, sps=8,
                            beta=0.35, snr_db=snr_db, target_rms=0.006,
                            cfo_std=0.0, rng=rng, fec=fec_mode,
                        )
                        bursts.append(b)

                    synth_iq_np = np.concatenate(
                        [b['iq_tensor'] for b in bursts], axis=0
                    )  # [N, 2, 128]

                    # Transfer adversarial perturbation
                    sample_idx = rng.integers(0, len(delta_real), size=n_bursts)
                    adv_synth_np = synth_iq_np + delta_real[sample_idx]

                    # Convert to torch for defense application
                    adv_synth_t = torch.from_numpy(adv_synth_np).float()

                    row = f"{mod_type:<8}{'Y' if fec_mode else 'N':>5}"

                    for def_name in defenses:
                        # Apply defense
                        defended_t = apply_defense_to_iq(def_name, adv_synth_t, device)
                        defended_np = defended_t.cpu().numpy()

                        # Demodulate each burst and check CRC
                        crc_pass = 0
                        for i in range(n_bursts):
                            # Convert defended IQ back to complex
                            defended_iq = (defended_np[i, 0, :]
                                           + 1j * defended_np[i, 1, :])
                            passed = demod_burst_with_defense(
                                bursts[i], defended_iq, mod_type, fec=fec_mode,
                            )
                            crc_pass += int(passed)

                        rate = crc_pass / n_bursts
                        row += f"{rate*100:>9.1f}%"

                    print(row)

            print()


if __name__ == '__main__':
    main()
