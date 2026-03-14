#!/usr/bin/env python
"""
Spectral-gated defense: ONE cheap spectral check + ONE defense + ONE inference.

Idea: compute a fast spectral feature from the raw signal (no model needed),
use it to select between two defenses per sample. Total cost:
  1 FFT (for feature) + 1 defense filter + 1 model inference = real-time.

Features to try:
1. Spectral flatness: geometric_mean(PSD) / arithmetic_mean(PSD)
   - Flat (noise-like, wideband) → Quant
   - Peaky (narrowband, tonal) → LP/Top-K

2. Low-band energy ratio: energy in bins 0-10 / total energy
   - High ratio → signal is narrowband, LP is safe
   - Low ratio → signal is wideband, use Quant

3. Spectral asymmetry: |left_energy - right_energy| / total
   - AM-SSB is asymmetric (single sideband)
"""
import numpy as np
import torch
import torch.nn.functional as F_torch
import pickle

from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial, compute_accuracy
from util.defense import fft_topk_denoise


def lowpass_filter(x, cutoff_bin=20):
    N, C, T = x.shape
    X = torch.fft.rfft(x, n=T, dim=2)
    mask = torch.zeros_like(X)
    mask[:, :, :cutoff_bin] = 1.0
    return torch.fft.irfft(X * mask, n=T, dim=2)

def input_quantize(x, n_levels=32):
    """Per-sample quantization (not per-batch)."""
    N = x.shape[0]
    result = x.clone()
    for i in range(N):
        xi = x[i]
        x_min = xi.min()
        x_max = xi.max()
        x_norm = (xi - x_min) / (x_max - x_min + 1e-12)
        x_quant = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
        result[i] = x_quant * (x_max - x_min) + x_min
    return result

def input_quantize_batch(x, n_levels=32):
    """Per-sample quantization, vectorized."""
    x_min = x.flatten(1).min(dim=1).values.view(-1, 1, 1)
    x_max = x.flatten(1).max(dim=1).values.view(-1, 1, 1)
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    x_quant = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
    return x_quant * (x_max - x_min) + x_min

def moving_avg_smooth(x, kernel_size=3):
    N, C, T = x.shape
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
    x_flat = x.view(N * C, 1, T)
    x_smooth = F_torch.conv1d(x_flat, kernel, padding=pad)
    return x_smooth.view(N, C, T)


IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}


def spectral_flatness(x):
    """Per-sample spectral flatness. Returns [N] tensor."""
    N, C, T = x.shape
    X = torch.fft.rfft(x, n=T, dim=2)
    power = X.abs() ** 2 + 1e-20  # [N, C, n_bins]
    log_power = torch.log(power)
    geo_mean = torch.exp(log_power.mean(dim=2))  # [N, C]
    arith_mean = power.mean(dim=2)  # [N, C]
    flatness = geo_mean / (arith_mean + 1e-12)  # [N, C]
    return flatness.mean(dim=1)  # [N]


def low_band_ratio(x, n_low=10):
    """Fraction of energy in first n_low FFT bins. Returns [N]."""
    N, C, T = x.shape
    X = torch.fft.rfft(x, n=T, dim=2)
    power = X.abs() ** 2
    total = power.sum(dim=2) + 1e-12
    low = power[:, :, :n_low].sum(dim=2)
    ratio = low / total  # [N, C]
    return ratio.max(dim=1).values  # [N] — max across I/Q


def spectral_gated_defense(x, threshold, feature_fn, def_a, def_b):
    """
    Per-sample defense selection based on spectral feature.
    feature > threshold → use def_a
    feature <= threshold → use def_b
    """
    feat = feature_fn(x)  # [N]
    use_a = feat > threshold
    use_b = ~use_a

    result = x.clone()
    if use_a.any():
        result[use_a] = def_a(x[use_a])
    if use_b.any():
        result[use_b] = def_b(x[use_b])

    return result, use_a, use_b


def main():
    fix_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
        rml_data = pickle.load(f, encoding='bytes')

    mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4',
            'AM-DSB', 'GFSK', 'CPFSK', 'AM-SSB', 'WBFM']
    n_samples = 200

    # ============================================================
    # Step 1: Examine spectral features on CLEAN data
    # ============================================================
    print("=" * 80)
    print("  Spectral Features on CLEAN data (for threshold calibration)")
    print("=" * 80)
    print(f"{'Mod':<8}{'SNR':>5}  {'Flatness':>10}  {'LowBand10':>10}  {'LowBand15':>10}")
    print("-" * 55)

    for snr in [0, 18]:
        for mod in mods:
            key = (mod.encode(), snr)
            if key not in rml_data:
                continue
            samples = rml_data[key][:n_samples]
            x = torch.from_numpy(samples).float().to(device)

            sf = spectral_flatness(x).mean().item()
            lb10 = low_band_ratio(x, n_low=10).mean().item()
            lb15 = low_band_ratio(x, n_low=15).mean().item()
            print(f"{mod:<8}{snr:>5}  {sf:>10.4f}  {lb10:>10.4f}  {lb15:>10.4f}")
        print()

    # ============================================================
    # Step 2: Test gated defenses
    # ============================================================
    attacks_to_test = ['eadl1', 'cw']

    # Defense options
    def_top20 = lambda x: fft_topk_denoise(x, topk=20)
    def_lp20 = lambda x: lowpass_filter(x, cutoff_bin=20)
    def_q32 = lambda x: input_quantize_batch(x, n_levels=32)
    def_q64 = lambda x: input_quantize_batch(x, n_levels=64)
    def_mva3 = lambda x: moving_avg_smooth(x, kernel_size=3)

    # Gated defense configs: (name, feature_fn, threshold, def_when_above, def_when_below)
    gated_configs = [
        # "If low-band energy high → signal narrowband → LP-20 safe; else Quant32"
        ('LB10>0.85→LP20|Q32', lambda x: low_band_ratio(x, 10), 0.85, def_lp20, def_q32),
        ('LB10>0.80→LP20|Q32', lambda x: low_band_ratio(x, 10), 0.80, def_lp20, def_q32),
        ('LB10>0.75→LP20|Q32', lambda x: low_band_ratio(x, 10), 0.75, def_lp20, def_q32),
        # Top-20 variant
        ('LB10>0.85→T20|Q32',  lambda x: low_band_ratio(x, 10), 0.85, def_top20, def_q32),
        ('LB10>0.80→T20|Q32',  lambda x: low_band_ratio(x, 10), 0.80, def_top20, def_q32),
        # MvAvg3 variant
        ('LB10>0.85→MvA3|Q32', lambda x: low_band_ratio(x, 10), 0.85, def_mva3, def_q32),
        # Flatness-based
        ('Flat>0.5→Q32|LP20',  spectral_flatness, 0.50, def_q32, def_lp20),
        ('Flat>0.4→Q32|LP20',  spectral_flatness, 0.40, def_q32, def_lp20),
        ('Flat>0.3→Q32|T20',   spectral_flatness, 0.30, def_q32, def_top20),
    ]

    for attack_name in attacks_to_test:
        attack = create_attack(attack_name, wrapped_model, cfg)

        for snr in [0, 18]:
            print(f"\n{'='*140}")
            print(f"  {attack_name.upper()} | SNR={snr}")
            print(f"{'='*140}")

            hdr = f"{'Mod':<8}{'Clean':>7}{'Att':>7}{'Top20':>8}{'LP20':>8}{'Q32':>8}"
            for gc in gated_configs:
                name = gc[0][:12]
                hdr += f"{name:>14}"
            print(hdr)
            print("-" * 140)

            all_accs = {gc[0]: [] for gc in gated_configs}
            all_accs.update({'top20': [], 'lp20': [], 'q32': []})

            for mod in mods:
                key = (mod.encode(), snr)
                if key not in rml_data:
                    continue
                samples = rml_data[key][:n_samples]
                x = torch.from_numpy(samples).float().to(device)
                true_idx = MOD_TO_IDX[mod]
                y = torch.full((len(x),), true_idx, dtype=torch.long, device=device)

                clean_acc = compute_accuracy(model, x, y)
                x_adv = generate_adversarial(
                    attack, x, y, wrapped_model=wrapped_model, ta_box='minmax',
                )
                att_acc = compute_accuracy(model, x_adv, y)

                t20_acc = compute_accuracy(model, def_top20(x_adv), y)
                lp20_acc = compute_accuracy(model, def_lp20(x_adv), y)
                q32_acc = compute_accuracy(model, def_q32(x_adv), y)
                all_accs['top20'].append(t20_acc)
                all_accs['lp20'].append(lp20_acc)
                all_accs['q32'].append(q32_acc)

                row = f"{mod:<8}{clean_acc*100:>6.1f}%{att_acc*100:>6.1f}%"
                row += f"{t20_acc*100:>7.1f}%{lp20_acc*100:>7.1f}%{q32_acc*100:>7.1f}%"

                for gc in gated_configs:
                    name, feat_fn, thresh, def_a, def_b = gc
                    x_def, mask_a, mask_b = spectral_gated_defense(
                        x_adv, thresh, feat_fn, def_a, def_b)
                    acc = compute_accuracy(model, x_def, y)
                    n_a = mask_a.sum().item()
                    row += f"{acc*100:>8.1f}%({n_a:>3})"
                    all_accs[name].append(acc)

                print(row)

            print("-" * 140)
            row = f"{'AVG':<8}{'':>7}{'':>7}"
            row += f"{np.mean(all_accs['top20'])*100:>7.1f}%"
            row += f"{np.mean(all_accs['lp20'])*100:>7.1f}%"
            row += f"{np.mean(all_accs['q32'])*100:>7.1f}%"
            for gc in gated_configs:
                avg = np.mean(all_accs[gc[0]])
                row += f"{avg*100:>13.1f}%"
            print(row)
            print()


if __name__ == '__main__':
    main()
