#!/usr/bin/env python
"""
Sweep concentration+distortion thresholds to find optimal settings.
Test on clean data at SNR=18 to see what K each modulation naturally gets.
"""
import pickle
import numpy as np
import torch
from collections import Counter

from util.utils import create_model, fix_seed
from util.config import Config
from util.defense import fft_topk_denoise
from util.adaptive_defense import concentration_distortion_topk_denoise


IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}


def compute_accuracy(model, x, y):
    with torch.no_grad():
        logits, _ = model(x)
        return (logits.argmax(1) == y).float().mean().item()


def main():
    fix_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = Config('2016.10a', train=False)
    cfg.device = device
    model = create_model(cfg, model_name='awn')
    model.load_state_dict(torch.load('./checkpoint/2016.10a_AWN.pkl', map_location=device))
    model.eval()

    with open('./data/RML2016.10a_dict.pkl', 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4',
            'AM-DSB', 'GFSK', 'CPFSK', 'AM-SSB', 'WBFM']

    # First: inspect raw concentration and distortion values per modulation
    print("=== Raw spectral stats on CLEAN data (SNR=18) ===")
    print(f"{'Mod':<8} {'C10':>6} {'C20':>6} {'C30':>6} {'D10':>6} {'D20':>6} {'D30':>6}")
    print("-" * 50)

    for mod in mods:
        key = (mod.encode(), 18)
        if key not in data:
            continue
        samples = data[key][:200]
        x = torch.from_numpy(samples).float().to(device)

        N, C, T = x.shape
        X = torch.fft.fft(x, n=T, dim=2)
        energy = X.abs() ** 2
        total_e = energy.sum(dim=2, keepdim=True)
        sorted_e, _ = energy.sort(dim=2, descending=True)
        x_power = (x ** 2).sum(dim=(1, 2)).clamp(min=1e-12)

        for k_val in [10, 20, 30]:
            topk_e = sorted_e[:, :, :k_val].sum(dim=2)
            conc = (topk_e / total_e.squeeze(2).clamp(min=1e-12)).max(dim=1).values
            x_topk = fft_topk_denoise(x, topk=k_val)
            diff = x - x_topk
            dist = (diff ** 2).sum(dim=(1, 2)) / x_power
            if k_val == 10:
                c10, d10 = conc.mean().item(), dist.mean().item()
            elif k_val == 20:
                c20, d20 = conc.mean().item(), dist.mean().item()
            else:
                c30, d30 = conc.mean().item(), dist.mean().item()

        print(f"{mod:<8} {c10:>6.3f} {c20:>6.3f} {c30:>6.3f} "
              f"{d10:>6.3f} {d20:>6.3f} {d30:>6.3f}")

    # Same for SNR=0
    print("\n=== Raw spectral stats on CLEAN data (SNR=0) ===")
    print(f"{'Mod':<8} {'C10':>6} {'C20':>6} {'C30':>6} {'D10':>6} {'D20':>6} {'D30':>6}")
    print("-" * 50)

    for mod in mods:
        key = (mod.encode(), 0)
        if key not in data:
            continue
        samples = data[key][:200]
        x = torch.from_numpy(samples).float().to(device)

        N, C, T = x.shape
        X = torch.fft.fft(x, n=T, dim=2)
        energy = X.abs() ** 2
        total_e = energy.sum(dim=2, keepdim=True)
        sorted_e, _ = energy.sort(dim=2, descending=True)
        x_power = (x ** 2).sum(dim=(1, 2)).clamp(min=1e-12)

        for k_val in [10, 20, 30]:
            topk_e = sorted_e[:, :, :k_val].sum(dim=2)
            conc = (topk_e / total_e.squeeze(2).clamp(min=1e-12)).max(dim=1).values
            x_topk = fft_topk_denoise(x, topk=k_val)
            diff = x - x_topk
            dist = (diff ** 2).sum(dim=(1, 2)) / x_power
            if k_val == 10:
                c10, d10 = conc.mean().item(), dist.mean().item()
            elif k_val == 20:
                c20, d20 = conc.mean().item(), dist.mean().item()
            else:
                c30, d30 = conc.mean().item(), dist.mean().item()

        print(f"{mod:<8} {c10:>6.3f} {c20:>6.3f} {c30:>6.3f} "
              f"{d10:>6.3f} {d20:>6.3f} {d30:>6.3f}")

    # Now test tuned thresholds
    # Based on data: if C10 > 0.70 AND D10 < 0.30 → K=10
    #                if C20 > 0.60 AND D20 < 0.40 → K=20
    #                if C30 > 0.50 AND D30 < 0.50 → K=30
    #                else reject
    print("\n=== Tuned thresholds test ===")
    configs = [
        ("strict",  [0.85, 0.80, 0.75], [0.15, 0.20, 0.25]),
        ("medium",  [0.70, 0.60, 0.50], [0.30, 0.40, 0.50]),
        ("relaxed", [0.55, 0.45, 0.40], [0.45, 0.55, 0.60]),
    ]

    for snr in [0, 18]:
        for config_name, conc_th, dist_th in configs:
            print(f"\n--- SNR={snr}, Config={config_name} "
                  f"(conc={conc_th}, dist={dist_th}) ---")
            print(f"{'Mod':<8} {'Clean':>7} {'CD_Acc':>7} {'K-dist':<35}")

            for mod in mods:
                key = (mod.encode(), snr)
                if key not in data:
                    continue
                samples = data[key][:200]
                x = torch.from_numpy(samples).float().to(device)
                true_idx = MOD_TO_IDX[mod]
                y = torch.full((len(x),), true_idx, dtype=torch.long, device=device)

                clean_acc = compute_accuracy(model, x, y)
                x_cd, sel_k = concentration_distortion_topk_denoise(
                    x, k_candidates=[10, 20, 30],
                    conc_thresholds=conc_th, dist_thresholds=dist_th,
                )
                cd_acc = compute_accuracy(model, x_cd, y)
                k_counts = Counter(sel_k.cpu().numpy().tolist())
                k_str = " ".join(f"K{int(k)}:{v}" for k, v in sorted(k_counts.items()))

                print(f"{mod:<8} {clean_acc*100:>6.1f}% {cd_acc*100:>6.1f}% {k_str:<35}")


if __name__ == '__main__':
    main()
