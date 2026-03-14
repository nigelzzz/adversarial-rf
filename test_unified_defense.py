#!/usr/bin/env python
"""
Test the unified spectral_gated_defense on all modulations, both attacks, both SNRs.
ONE defense. ONE inference. Real-time.
"""
import time
import numpy as np
import torch
import pickle

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
    # Speed benchmark
    # ============================================================
    print("=" * 60)
    print("  SPEED BENCHMARK")
    print("=" * 60)
    x_dummy = torch.randn(200, 2, 128, device=device) * 0.01
    for name, fn in [
        ('Top-20', lambda x: fft_topk_denoise(x, topk=20)),
        ('Gated', lambda x: spectral_gated_defense(x)),
    ]:
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        for _ in range(50):
            _ = fn(x_dummy)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        ms = (time.time() - t0) / 50 * 1000
        print(f"  {name:<10}: {ms:.2f} ms / batch(200)")

    # ============================================================
    # Clean accuracy (no attack)
    # ============================================================
    print(f"\n{'='*80}")
    print(f"  CLEAN ACCURACY (no attack)")
    print(f"{'='*80}")
    print(f"{'Mod':<8}{'SNR':>5}{'Raw':>8}{'Top-20':>8}{'Gated':>8}{'Route':>8}")
    print("-" * 45)

    for snr in [0, 18]:
        for mod in mods:
            key = (mod.encode(), snr)
            if key not in rml_data:
                continue
            samples = rml_data[key][:n_samples]
            x = torch.from_numpy(samples).float().to(device)
            true_idx = MOD_TO_IDX[mod]
            y = torch.full((len(x),), true_idx, dtype=torch.long, device=device)

            raw = compute_accuracy(model, x, y)
            t20 = compute_accuracy(model, fft_topk_denoise(x, topk=20), y)
            gated = compute_accuracy(model, spectral_gated_defense(x), y)

            # Check routing
            X = torch.fft.rfft(x, n=128, dim=2)
            power = X.abs() ** 2 + 1e-20
            geo = torch.exp(torch.log(power).mean(dim=2))
            arith = power.mean(dim=2)
            flat = (geo / (arith + 1e-12)).mean(dim=1)
            n_wide = (flat > 0.4).sum().item()
            route = f"Q:{n_wide}" if n_wide > 0 else "T:all"

            print(f"{mod:<8}{snr:>5}{raw*100:>7.1f}%{t20*100:>7.1f}%{gated*100:>7.1f}%{route:>8}")
        print()

    # ============================================================
    # Attack recovery
    # ============================================================
    for attack_name in ['eadl1', 'cw']:
        attack = create_attack(attack_name, wrapped_model, cfg)

        for snr in [0, 18]:
            print(f"{'='*80}")
            print(f"  {attack_name.upper()} | SNR={snr}")
            print(f"{'='*80}")
            print(f"{'Mod':<8}{'Clean':>7}{'Att':>7}{'Top-20':>8}{'Gated':>8}{'Diff':>7}{'Route':>8}")
            print("-" * 55)

            sum_clean = sum_att = sum_t20 = sum_gated = 0
            n_mods = 0

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

                t20_acc = compute_accuracy(model, fft_topk_denoise(x_adv, topk=20), y)
                gated_x = spectral_gated_defense(x_adv)
                gated_acc = compute_accuracy(model, gated_x, y)

                diff = gated_acc - t20_acc

                # Check routing
                X = torch.fft.rfft(x_adv, n=128, dim=2)
                power = X.abs() ** 2 + 1e-20
                geo = torch.exp(torch.log(power).mean(dim=2))
                arith = power.mean(dim=2)
                flat = (geo / (arith + 1e-12)).mean(dim=1)
                n_wide = (flat > 0.4).sum().item()
                route = f"Q:{n_wide}" if n_wide > 0 else "T:all"

                sign = "+" if diff > 0 else ""
                print(f"{mod:<8}{clean_acc*100:>6.1f}%{att_acc*100:>6.1f}%"
                      f"{t20_acc*100:>7.1f}%{gated_acc*100:>7.1f}%"
                      f"{sign}{diff*100:>5.1f}%{route:>8}")

                sum_clean += clean_acc
                sum_att += att_acc
                sum_t20 += t20_acc
                sum_gated += gated_acc
                n_mods += 1

            print("-" * 55)
            print(f"{'AVG':<8}{sum_clean/n_mods*100:>6.1f}%{sum_att/n_mods*100:>6.1f}%"
                  f"{sum_t20/n_mods*100:>7.1f}%{sum_gated/n_mods*100:>7.1f}%"
                  f"{'+' if (sum_gated-sum_t20)>0 else ''}"
                  f"{(sum_gated-sum_t20)/n_mods*100:>5.1f}%")
            print()


if __name__ == '__main__':
    main()
