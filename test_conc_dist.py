#!/usr/bin/env python
"""
Test concentration+distortion adaptive Top-K on all modulations at SNR 0 and 18.

Reports: clean acc, attack acc (CW, EADL1), and conc_dist defense acc per modulation.
Also reports K distribution to understand what K each modulation gets.
"""
import os
import pickle
import numpy as np
import torch
from collections import Counter

from util.utils import create_model, fix_seed
from util.config import Config
from util.adv_attack import Model01Wrapper
from util.sigguard_eval import create_attack, generate_adversarial, compute_accuracy
from util.defense import fft_topk_denoise
from util.adaptive_defense import concentration_distortion_topk_denoise


IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}


def main():
    fix_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    cfg = Config('2016.10a', train=False)
    cfg.device = device
    model = create_model(cfg, model_name='awn')
    model.load_state_dict(torch.load('./checkpoint/2016.10a_AWN.pkl', map_location=device))
    model.eval()

    # Attack config
    cfg.cw_c = 10.0
    cfg.cw_steps = 200
    cfg.cw_lr = 0.005
    cfg.attack_eps = 0.1
    cfg.ta_box = 'minmax'
    cfg.num_classes = 11

    wrapped_model = Model01Wrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()

    # Load dataset
    with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    attacks_to_test = ['cw', 'eadl1']
    snrs_to_test = [0, 18]
    mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4',
            'AM-DSB', 'GFSK', 'CPFSK', 'AM-SSB', 'WBFM']
    n_samples = 200

    # Print header
    print(f"{'Mod':<8} {'SNR':>4} {'Attack':<8} {'Clean':>7} {'Att':>7} "
          f"{'T10':>7} {'T20':>7} {'CD':>7} {'K-dist':<30}")
    print("-" * 105)

    for snr in snrs_to_test:
        for attack_name in attacks_to_test:
            attack = create_attack(attack_name, wrapped_model, cfg)

            for mod in mods:
                key = (mod.encode(), snr)
                if key not in data:
                    continue
                samples = data[key]
                if n_samples < len(samples):
                    idx = np.random.choice(len(samples), n_samples, replace=False)
                    samples = samples[idx]

                x = torch.from_numpy(samples).float().to(device)
                true_idx = MOD_TO_IDX[mod]
                y = torch.full((len(x),), true_idx, dtype=torch.long, device=device)

                # Clean accuracy
                clean_acc = compute_accuracy(model, x, y)

                # Attack
                x_adv = generate_adversarial(
                    attack, x, y,
                    wrapped_model=wrapped_model,
                    ta_box='minmax',
                )
                att_acc = compute_accuracy(model, x_adv, y)

                # Fixed Top-K
                x_t10 = fft_topk_denoise(x_adv, topk=10)
                x_t20 = fft_topk_denoise(x_adv, topk=20)
                t10_acc = compute_accuracy(model, x_t10, y)
                t20_acc = compute_accuracy(model, x_t20, y)

                # Concentration+Distortion adaptive
                x_cd, sel_k = concentration_distortion_topk_denoise(x_adv)
                cd_acc = compute_accuracy(model, x_cd, y)

                # K distribution
                k_counts = Counter(sel_k.cpu().numpy().tolist())
                k_str = " ".join(f"K{int(k)}:{v}" for k, v in sorted(k_counts.items()))

                print(f"{mod:<8} {snr:>4} {attack_name:<8} "
                      f"{clean_acc*100:>6.1f}% {att_acc*100:>6.1f}% "
                      f"{t10_acc*100:>6.1f}% {t20_acc*100:>6.1f}% "
                      f"{cd_acc*100:>6.1f}% {k_str:<30}")

        print()


if __name__ == '__main__':
    main()
