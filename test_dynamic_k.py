#!/usr/bin/env python
"""
Dynamic K selection for FFT Top-K defense at SNR=18.

Approaches:
1. Three-tier routing (spectral features, no model):
   - flatness > 0.4        → Quant32       (AM-SSB)
   - low_band_ratio > 0.9  → Top-10        (QAM64, PAM4, AM-DSB)
   - else                  → Top-20        (BPSK, QPSK, 8PSK, ...)

2. Continuous K via occupied bandwidth:
   - Estimate occupied BW (bins covering 99% energy)
   - Map BW → K linearly: K = clamp(bw * scale, 10, 50)

3. Current spectral_gated_defense (baseline: flatness → Quant or Top-20)

Compare per-modulation accuracy for CW and EADEN attacks.
"""
import numpy as np
import torch
import pickle
from collections import Counter

from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial, compute_accuracy
from util.defense import fft_topk_denoise, spectral_gated_defense


IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}


def _per_sample_quantize(x, n_levels=32):
    x_flat = x.flatten(1)
    x_min = x_flat.min(dim=1).values.view(-1, 1, 1)
    x_max = x_flat.max(dim=1).values.view(-1, 1, 1)
    rng = x_max - x_min + 1e-12
    x_norm = (x - x_min) / rng
    x_quant = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
    return x_quant * rng + x_min


def spectral_flatness(x):
    N, C, T = x.shape
    X = torch.fft.rfft(x, n=T, dim=2)
    power = X.abs() ** 2 + 1e-20
    geo = torch.exp(torch.log(power).mean(dim=2))
    arith = power.mean(dim=2)
    return (geo / (arith + 1e-12)).mean(dim=1)


def low_band_ratio(x, n_low=10):
    N, C, T = x.shape
    X = torch.fft.rfft(x, n=T, dim=2)
    power = X.abs() ** 2
    total = power.sum(dim=2) + 1e-12
    low = power[:, :, :n_low].sum(dim=2)
    return (low / total).max(dim=1).values


def occupied_bandwidth(x, fraction=0.99):
    """Per-sample occupied bandwidth: number of FFT bins covering `fraction` of energy."""
    N, C, T = x.shape
    X = torch.fft.fft(x, n=T, dim=2)
    energy = X.abs() ** 2
    sorted_e, _ = energy.sort(dim=2, descending=True)
    cumsum = sorted_e.cumsum(dim=2)
    total = energy.sum(dim=2, keepdim=True)
    reached = cumsum >= fraction * total
    bw_per_ch = reached.float().argmax(dim=2) + 1
    never = ~reached.any(dim=2)
    bw_per_ch[never] = T
    return bw_per_ch.max(dim=1).values.float()


# ---- Defense strategies ----

def defense_fixed(x, k):
    return fft_topk_denoise(x, topk=k), f"K={k}"


def defense_three_tier(x, flatness_th=0.4, lb_th=0.90, n_low=10):
    """Three-tier: Quant32 / Top-10 / Top-20."""
    N, C, T = x.shape
    flat = spectral_flatness(x)
    lb = low_band_ratio(x, n_low=n_low)

    is_wideband = flat > flatness_th
    is_narrowband = (~is_wideband) & (lb > lb_th)
    is_mid = ~is_wideband & ~is_narrowband

    result = x.clone()
    k_used = torch.full((N,), 20, dtype=torch.long, device=x.device)

    if is_wideband.any():
        result[is_wideband] = _per_sample_quantize(x[is_wideband], 32)
        k_used[is_wideband] = 0  # 0 = quant

    if is_narrowband.any():
        result[is_narrowband] = fft_topk_denoise(x[is_narrowband], topk=10)
        k_used[is_narrowband] = 10

    if is_mid.any():
        result[is_mid] = fft_topk_denoise(x[is_mid], topk=20)
        k_used[is_mid] = 20

    dist = Counter(k_used.cpu().tolist())
    return result, f"Q:{dist.get(0,0)} K10:{dist.get(10,0)} K20:{dist.get(20,0)}"


def defense_four_tier(x, flatness_th=0.4, lb_th_high=0.92, lb_th_mid=0.80, n_low=10):
    """Four-tier: Quant32 / Top-10 / Top-20 / Top-30."""
    N, C, T = x.shape
    flat = spectral_flatness(x)
    lb = low_band_ratio(x, n_low=n_low)

    is_wideband = flat > flatness_th
    is_narrow = (~is_wideband) & (lb > lb_th_high)
    is_mid = (~is_wideband) & (~is_narrow) & (lb > lb_th_mid)
    is_wide_digital = ~is_wideband & ~is_narrow & ~is_mid

    result = x.clone()
    k_used = torch.full((N,), 30, dtype=torch.long, device=x.device)

    if is_wideband.any():
        result[is_wideband] = _per_sample_quantize(x[is_wideband], 32)
        k_used[is_wideband] = 0
    if is_narrow.any():
        result[is_narrow] = fft_topk_denoise(x[is_narrow], topk=10)
        k_used[is_narrow] = 10
    if is_mid.any():
        result[is_mid] = fft_topk_denoise(x[is_mid], topk=20)
        k_used[is_mid] = 20
    if is_wide_digital.any():
        result[is_wide_digital] = fft_topk_denoise(x[is_wide_digital], topk=30)
        k_used[is_wide_digital] = 30

    dist = Counter(k_used.cpu().tolist())
    return result, f"Q:{dist.get(0,0)} K10:{dist.get(10,0)} K20:{dist.get(20,0)} K30:{dist.get(30,0)}"


def defense_continuous_bw(x, fraction=0.99, k_min=10, k_max=50, scale=1.5):
    """Map occupied bandwidth → K continuously, then per-sample Top-K."""
    N, C, T = x.shape
    flat = spectral_flatness(x)
    is_wideband = flat > 0.4

    bw = occupied_bandwidth(x, fraction=fraction)
    k_per_sample = (bw * scale).clamp(k_min, k_max).long()

    # Wideband override → quant
    result = x.clone()
    if is_wideband.any():
        result[is_wideband] = _per_sample_quantize(x[is_wideband], 32)
        k_per_sample[is_wideband] = 0

    # Group by K for efficiency
    unique_ks = k_per_sample[~is_wideband].unique()
    for k in unique_ks:
        mask = (k_per_sample == k) & ~is_wideband
        if mask.any():
            result[mask] = fft_topk_denoise(x[mask], topk=k.item())

    dist = Counter(k_per_sample.cpu().tolist())
    return result, str(dict(sorted(dist.items())))


def defense_continuous_bw_v2(x, fraction=0.95, k_min=10, k_max=40, scale=2.0):
    """Variant: 95% energy, scale=2.0, max=40."""
    N, C, T = x.shape
    flat = spectral_flatness(x)
    is_wideband = flat > 0.4

    bw = occupied_bandwidth(x, fraction=fraction)
    k_per_sample = (bw * scale).clamp(k_min, k_max).long()

    result = x.clone()
    if is_wideband.any():
        result[is_wideband] = _per_sample_quantize(x[is_wideband], 32)
        k_per_sample[is_wideband] = 0

    unique_ks = k_per_sample[~is_wideband].unique()
    for k in unique_ks:
        mask = (k_per_sample == k) & ~is_wideband
        if mask.any():
            result[mask] = fft_topk_denoise(x[mask], topk=k.item())

    dist = Counter(k_per_sample.cpu().tolist())
    return result, str(dict(sorted(dist.items())))


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
    snr = 18

    # ================================================================
    # Step 1: Profile spectral features on clean data
    # ================================================================
    print("=" * 90)
    print("  SPECTRAL PROFILE (clean, SNR=18)")
    print("=" * 90)
    print(f"{'Mod':<8} {'Flatness':>9} {'LB10':>7} {'LB15':>7} {'OccBW99':>8} {'OccBW95':>8}")
    print("-" * 52)

    for mod in mods:
        key = (mod.encode(), snr)
        if key not in rml_data:
            continue
        x = torch.from_numpy(rml_data[key][:n_samples]).float().to(device)
        sf = spectral_flatness(x).mean().item()
        lb10 = low_band_ratio(x, 10).mean().item()
        lb15 = low_band_ratio(x, 15).mean().item()
        bw99 = occupied_bandwidth(x, 0.99).mean().item()
        bw95 = occupied_bandwidth(x, 0.95).mean().item()
        print(f"{mod:<8} {sf:>9.4f} {lb10:>7.4f} {lb15:>7.4f} {bw99:>8.1f} {bw95:>8.1f}")

    # ================================================================
    # Step 2: Clean accuracy impact
    # ================================================================
    print(f"\n{'='*110}")
    print(f"  CLEAN ACCURACY (SNR=18) — defense cost on unattacked signals")
    print(f"{'='*110}")
    print(f"{'Mod':<8} {'Raw':>6} {'T10':>6} {'T20':>6} {'T50':>6} {'Gate2':>6} {'3Tier':>6} {'4Tier':>6} {'BW99':>6} {'BW95':>6}")
    print("-" * 70)

    clean_accs = {d: [] for d in ['raw', 't10', 't20', 't50', 'gate2', '3tier', '4tier', 'bw99', 'bw95']}

    for mod in mods:
        key = (mod.encode(), snr)
        if key not in rml_data:
            continue
        x = torch.from_numpy(rml_data[key][:n_samples]).float().to(device)
        y = torch.full((len(x),), MOD_TO_IDX[mod], dtype=torch.long, device=device)

        raw = compute_accuracy(model, x, y)
        t10 = compute_accuracy(model, fft_topk_denoise(x, 10), y)
        t20 = compute_accuracy(model, fft_topk_denoise(x, 20), y)
        t50 = compute_accuracy(model, fft_topk_denoise(x, 50), y)
        gate2 = compute_accuracy(model, spectral_gated_defense(x), y)
        x_3t, _ = defense_three_tier(x)
        tier3 = compute_accuracy(model, x_3t, y)
        x_4t, _ = defense_four_tier(x)
        tier4 = compute_accuracy(model, x_4t, y)
        x_bw99, _ = defense_continuous_bw(x)
        bw99 = compute_accuracy(model, x_bw99, y)
        x_bw95, _ = defense_continuous_bw_v2(x)
        bw95 = compute_accuracy(model, x_bw95, y)

        for k, v in [('raw',raw),('t10',t10),('t20',t20),('t50',t50),
                     ('gate2',gate2),('3tier',tier3),('4tier',tier4),
                     ('bw99',bw99),('bw95',bw95)]:
            clean_accs[k].append(v)

        print(f"{mod:<8} {raw*100:>5.1f}% {t10*100:>5.1f}% {t20*100:>5.1f}% {t50*100:>5.1f}% "
              f"{gate2*100:>5.1f}% {tier3*100:>5.1f}% {tier4*100:>5.1f}% {bw99*100:>5.1f}% {bw95*100:>5.1f}%")

    print("-" * 70)
    print(f"{'AVG':<8}", end="")
    for k in ['raw','t10','t20','t50','gate2','3tier','4tier','bw99','bw95']:
        print(f" {np.mean(clean_accs[k])*100:>5.1f}%", end="")
    print()

    # ================================================================
    # Step 3: Attack recovery
    # ================================================================
    for attack_name in ['cw', 'eaden']:
        attack = create_attack(attack_name, wrapped_model, cfg)

        print(f"\n{'='*130}")
        print(f"  {attack_name.upper()} ATTACK | SNR={snr}")
        print(f"{'='*130}")
        print(f"{'Mod':<8} {'Clean':>6} {'Att':>6} {'T10':>6} {'T20':>6} {'T50':>6} "
              f"{'Gate2':>6} {'3Tier':>6} {'4Tier':>6} {'BW99':>6} {'BW95':>6}  {'Route (3Tier)'}")
        print("-" * 130)

        atk_accs = {d: [] for d in ['att', 't10', 't20', 't50', 'gate2', '3tier', '4tier', 'bw99', 'bw95']}

        for mod in mods:
            key = (mod.encode(), snr)
            if key not in rml_data:
                continue
            x = torch.from_numpy(rml_data[key][:n_samples]).float().to(device)
            y = torch.full((len(x),), MOD_TO_IDX[mod], dtype=torch.long, device=device)

            clean_acc = compute_accuracy(model, x, y)
            x_adv = generate_adversarial(
                attack, x, y, wrapped_model=wrapped_model, ta_box='minmax',
            )
            att_acc = compute_accuracy(model, x_adv, y)

            t10 = compute_accuracy(model, fft_topk_denoise(x_adv, 10), y)
            t20 = compute_accuracy(model, fft_topk_denoise(x_adv, 20), y)
            t50 = compute_accuracy(model, fft_topk_denoise(x_adv, 50), y)
            gate2 = compute_accuracy(model, spectral_gated_defense(x_adv), y)
            x_3t, route3 = defense_three_tier(x_adv)
            tier3 = compute_accuracy(model, x_3t, y)
            x_4t, route4 = defense_four_tier(x_adv)
            tier4 = compute_accuracy(model, x_4t, y)
            x_bw99, rbw99 = defense_continuous_bw(x_adv)
            bw99 = compute_accuracy(model, x_bw99, y)
            x_bw95, rbw95 = defense_continuous_bw_v2(x_adv)
            bw95 = compute_accuracy(model, x_bw95, y)

            for k, v in [('att',att_acc),('t10',t10),('t20',t20),('t50',t50),
                         ('gate2',gate2),('3tier',tier3),('4tier',tier4),
                         ('bw99',bw99),('bw95',bw95)]:
                atk_accs[k].append(v)

            print(f"{mod:<8} {clean_acc*100:>5.1f}% {att_acc*100:>5.1f}% "
                  f"{t10*100:>5.1f}% {t20*100:>5.1f}% {t50*100:>5.1f}% "
                  f"{gate2*100:>5.1f}% {tier3*100:>5.1f}% {tier4*100:>5.1f}% "
                  f"{bw99*100:>5.1f}% {bw95*100:>5.1f}%  {route3}")

        print("-" * 130)
        print(f"{'AVG':<8} {'':>6} {'':>6}", end="")
        for k in ['t10','t20','t50','gate2','3tier','4tier','bw99','bw95']:
            print(f" {np.mean(atk_accs[k])*100:>5.1f}%", end="")
        print()


if __name__ == '__main__':
    main()
