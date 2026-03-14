#!/usr/bin/env python
"""
Multi-K Voting adaptive defense.

Algorithm:
  1. Apply Top-10 and Top-20 to the input
  2. Classify both filtered versions
  3. If predictions AGREE → use Top-10 (more aggressive filtering is safe)
  4. If predictions DISAGREE → Top-10 distorted the signal → use Top-20
  5. Optional: extend to K=30 as third tier

No thresholds, no calibration, no spectral analysis needed.
The classifier itself tells us whether aggressive filtering is safe.

Test on all modulations at SNR 0 and 18 with CW and EADL1 attacks.
"""
import pickle
import numpy as np
import torch
from collections import Counter

from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial, compute_accuracy
from util.defense import fft_topk_denoise


IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}


@torch.no_grad()
def multik_vote_denoise(x, model, k_low=10, k_high=20):
    """
    Multi-K voting: apply two K values, use agreement to decide.

    If Top-k_low and Top-k_high predictions agree → use Top-k_low (aggressive).
    If they disagree → use Top-k_high (conservative).

    Returns: (x_denoised, selected_k, pred_low, pred_high)
    """
    model.eval()
    N = x.shape[0]
    device = x.device

    x_low = fft_topk_denoise(x, topk=k_low)
    x_high = fft_topk_denoise(x, topk=k_high)

    logits_low, _ = model(x_low)
    logits_high, _ = model(x_high)

    pred_low = logits_low.argmax(dim=1)
    pred_high = logits_high.argmax(dim=1)

    agree = (pred_low == pred_high)

    result = x_high.clone()
    result[agree] = x_low[agree]

    selected_k = torch.full((N,), k_high, dtype=torch.int, device=device)
    selected_k[agree] = k_low

    return result, selected_k


@torch.no_grad()
def multik_vote_3tier(x, model, k1=10, k2=20, k3=30):
    """
    3-tier voting: K=10 vs K=20 vs K=30.

    - If pred_10 == pred_20 → use K=10
    - Elif pred_20 == pred_30 → use K=20
    - Else → use K=30
    """
    model.eval()
    N = x.shape[0]
    device = x.device

    x1 = fft_topk_denoise(x, topk=k1)
    x2 = fft_topk_denoise(x, topk=k2)
    x3 = fft_topk_denoise(x, topk=k3)

    logits1, _ = model(x1)
    logits2, _ = model(x2)
    logits3, _ = model(x3)

    p1 = logits1.argmax(dim=1)
    p2 = logits2.argmax(dim=1)
    p3 = logits3.argmax(dim=1)

    # Default: K=30
    result = x3.clone()
    selected_k = torch.full((N,), k3, dtype=torch.int, device=device)

    # Tier 2: if p2 == p3 → use K=20
    agree23 = (p2 == p3)
    result[agree23] = x2[agree23]
    selected_k[agree23] = k2

    # Tier 1: if p1 == p2 → use K=10 (overrides tier 2)
    agree12 = (p1 == p2)
    result[agree12] = x1[agree12]
    selected_k[agree12] = k1

    return result, selected_k


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
        data = pickle.load(f, encoding='bytes')

    attacks_to_test = ['cw', 'eadl1']
    snrs_to_test = [0, 18]
    mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4',
            'AM-DSB', 'GFSK', 'CPFSK', 'AM-SSB', 'WBFM']
    n_samples = 200

    # === Test 1: Clean accuracy (no attack) ===
    print("=" * 110)
    print("  CLEAN ACCURACY (no attack)")
    print("=" * 110)
    print(f"{'Mod':<8} {'SNR':>4} {'Raw':>7} {'T10':>7} {'T20':>7} {'T30':>7} "
          f"{'Vote2':>7} {'Vote3':>7} {'V2-Kdist':<20} {'V3-Kdist':<20}")
    print("-" * 110)

    for snr in snrs_to_test:
        for mod in mods:
            key = (mod.encode(), snr)
            if key not in data:
                continue
            samples = data[key][:n_samples]
            x = torch.from_numpy(samples).float().to(device)
            true_idx = MOD_TO_IDX[mod]
            y = torch.full((len(x),), true_idx, dtype=torch.long, device=device)

            raw_acc = compute_accuracy(model, x, y)
            t10_acc = compute_accuracy(model, fft_topk_denoise(x, topk=10), y)
            t20_acc = compute_accuracy(model, fft_topk_denoise(x, topk=20), y)
            t30_acc = compute_accuracy(model, fft_topk_denoise(x, topk=30), y)

            x_v2, sk2 = multik_vote_denoise(x, model)
            v2_acc = compute_accuracy(model, x_v2, y)
            kd2 = Counter(sk2.cpu().numpy().tolist())
            kd2_str = " ".join(f"K{int(k)}:{v}" for k, v in sorted(kd2.items()))

            x_v3, sk3 = multik_vote_3tier(x, model)
            v3_acc = compute_accuracy(model, x_v3, y)
            kd3 = Counter(sk3.cpu().numpy().tolist())
            kd3_str = " ".join(f"K{int(k)}:{v}" for k, v in sorted(kd3.items()))

            print(f"{mod:<8} {snr:>4} {raw_acc*100:>6.1f}% {t10_acc*100:>6.1f}% "
                  f"{t20_acc*100:>6.1f}% {t30_acc*100:>6.1f}% "
                  f"{v2_acc*100:>6.1f}% {v3_acc*100:>6.1f}% "
                  f"{kd2_str:<20} {kd3_str:<20}")
        print()

    # === Test 2: Under attack ===
    for attack_name in attacks_to_test:
        attack = create_attack(attack_name, wrapped_model, cfg)

        print("=" * 110)
        print(f"  ATTACK: {attack_name.upper()}")
        print("=" * 110)
        print(f"{'Mod':<8} {'SNR':>4} {'Clean':>7} {'Att':>7} {'T10':>7} {'T20':>7} "
              f"{'Vote2':>7} {'Vote3':>7} {'V2-Kdist':<20} {'V3-Kdist':<20}")
        print("-" * 110)

        for snr in snrs_to_test:
            for mod in mods:
                key = (mod.encode(), snr)
                if key not in data:
                    continue
                samples = data[key][:n_samples]
                x = torch.from_numpy(samples).float().to(device)
                true_idx = MOD_TO_IDX[mod]
                y = torch.full((len(x),), true_idx, dtype=torch.long, device=device)

                clean_acc = compute_accuracy(model, x, y)

                x_adv = generate_adversarial(
                    attack, x, y,
                    wrapped_model=wrapped_model,
                    ta_box='minmax',
                )
                att_acc = compute_accuracy(model, x_adv, y)
                t10_acc = compute_accuracy(model, fft_topk_denoise(x_adv, topk=10), y)
                t20_acc = compute_accuracy(model, fft_topk_denoise(x_adv, topk=20), y)

                x_v2, sk2 = multik_vote_denoise(x_adv, model)
                v2_acc = compute_accuracy(model, x_v2, y)
                kd2 = Counter(sk2.cpu().numpy().tolist())
                kd2_str = " ".join(f"K{int(k)}:{v}" for k, v in sorted(kd2.items()))

                x_v3, sk3 = multik_vote_3tier(x_adv, model)
                v3_acc = compute_accuracy(model, x_v3, y)
                kd3 = Counter(sk3.cpu().numpy().tolist())
                kd3_str = " ".join(f"K{int(k)}:{v}" for k, v in sorted(kd3.items()))

                print(f"{mod:<8} {snr:>4} {clean_acc*100:>6.1f}% {att_acc*100:>6.1f}% "
                      f"{t10_acc*100:>6.1f}% {t20_acc*100:>6.1f}% "
                      f"{v2_acc*100:>6.1f}% {v3_acc*100:>6.1f}% "
                      f"{kd2_str:<20} {kd3_str:<20}")
            print()


if __name__ == '__main__':
    main()
