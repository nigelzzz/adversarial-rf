#!/usr/bin/env python
"""
Test alternative defense methods beyond FFT Top-K.

1. Moving average smoothing (1D conv) — O(N), no FFT
2. Gaussian smoothing — O(N)
3. Median filter — robust to impulse noise
4. Input quantization — destroy LSB perturbations
5. Additive noise defense — randomize input
6. Bit-depth reduction — coarsen signal
7. Adversarial denoising via small AE (if trained)

Compare: speed (ms per batch) and accuracy recovery.
"""
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
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


# ============================================================
# Defense functions
# ============================================================

def moving_avg_smooth(x, kernel_size=5):
    """1D moving average per channel. [N,2,T] → [N,2,T]."""
    N, C, T = x.shape
    pad = kernel_size // 2
    # Use conv1d with uniform kernel
    kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
    x_flat = x.view(N * C, 1, T)
    x_smooth = F.conv1d(x_flat, kernel, padding=pad)
    return x_smooth.view(N, C, T)


def gaussian_smooth(x, kernel_size=5, sigma=1.0):
    """1D Gaussian smoothing per channel."""
    N, C, T = x.shape
    pad = kernel_size // 2
    # Build Gaussian kernel
    t = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - pad
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    x_flat = x.view(N * C, 1, T)
    x_smooth = F.conv1d(x_flat, kernel, padding=pad)
    return x_smooth.view(N, C, T)


def median_filter(x, kernel_size=5):
    """1D median filter per channel."""
    N, C, T = x.shape
    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    # Unfold to get sliding windows
    windows = x_padded.unfold(dimension=2, size=kernel_size, step=1)  # [N,C,T,K]
    return windows.median(dim=3).values


def input_quantize(x, n_levels=32):
    """Quantize input to n_levels discrete values."""
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    x_quant = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
    return x_quant * (x_max - x_min) + x_min


def additive_noise_defense(x, model, noise_std=0.002, n_votes=5):
    """Add random noise, classify multiple times, majority vote."""
    model.eval()
    N = x.shape[0]
    all_preds = []
    with torch.no_grad():
        for _ in range(n_votes):
            x_noisy = x + torch.randn_like(x) * noise_std
            logits, _ = model(x_noisy)
            all_preds.append(logits.argmax(dim=1))
    # Stack and take mode
    preds = torch.stack(all_preds, dim=0)  # [n_votes, N]
    # majority vote
    final_preds = torch.mode(preds, dim=0).values
    return final_preds


def bit_depth_reduce(x, bits=8):
    """Reduce bit depth of signal values."""
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    levels = 2 ** bits
    x_reduced = torch.round(x_norm * (levels - 1)) / (levels - 1)
    return x_reduced * (x_max - x_min) + x_min


def lowpass_filter(x, cutoff_bin=20):
    """Simple FFT lowpass: zero bins above cutoff."""
    N, C, T = x.shape
    X = torch.fft.rfft(x, n=T, dim=2)
    mask = torch.zeros_like(X)
    mask[:, :, :cutoff_bin] = 1.0
    X_filtered = X * mask
    return torch.fft.irfft(X_filtered, n=T, dim=2)


@torch.no_grad()
def multik_vote_denoise(x, model, k_low=10, k_high=20):
    """Multi-K vote (2-tier)."""
    model.eval()
    x_low = fft_topk_denoise(x, topk=k_low)
    x_high = fft_topk_denoise(x, topk=k_high)
    logits_low, _ = model(x_low)
    logits_high, _ = model(x_high)
    agree = (logits_low.argmax(1) == logits_high.argmax(1))
    result = x_high.clone()
    result[agree] = x_low[agree]
    return result


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

    mods = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'PAM4',
            'AM-DSB', 'GFSK', 'CPFSK', 'AM-SSB', 'WBFM']
    n_samples = 200

    # Define defenses to test
    defenses = {
        'Top-10': lambda x: fft_topk_denoise(x, topk=10),
        'Top-20': lambda x: fft_topk_denoise(x, topk=20),
        'Vote2': lambda x: multik_vote_denoise(x, model),
        'MvAvg3': lambda x: moving_avg_smooth(x, kernel_size=3),
        'MvAvg5': lambda x: moving_avg_smooth(x, kernel_size=5),
        'Gauss3': lambda x: gaussian_smooth(x, kernel_size=5, sigma=1.0),
        'Gauss5': lambda x: gaussian_smooth(x, kernel_size=7, sigma=2.0),
        'Median3': lambda x: median_filter(x, kernel_size=3),
        'Median5': lambda x: median_filter(x, kernel_size=5),
        'Quant16': lambda x: input_quantize(x, n_levels=16),
        'Quant32': lambda x: input_quantize(x, n_levels=32),
        'Quant64': lambda x: input_quantize(x, n_levels=64),
        'BitR-6': lambda x: bit_depth_reduce(x, bits=6),
        'BitR-8': lambda x: bit_depth_reduce(x, bits=8),
        'LP-15': lambda x: lowpass_filter(x, cutoff_bin=15),
        'LP-20': lambda x: lowpass_filter(x, cutoff_bin=20),
        'LP-30': lambda x: lowpass_filter(x, cutoff_bin=30),
    }

    # ============================================================
    # Speed benchmark
    # ============================================================
    print("=" * 60)
    print("  SPEED BENCHMARK (200 samples, ms per batch)")
    print("=" * 60)
    # Use a dummy batch for timing
    x_dummy = torch.randn(200, 2, 128, device=device) * 0.01
    for name, defense_fn in defenses.items():
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t0 = time.time()
        for _ in range(10):
            _ = defense_fn(x_dummy)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        t1 = time.time()
        ms = (t1 - t0) / 10 * 1000
        print(f"  {name:<10}: {ms:>6.2f} ms")

    # ============================================================
    # Attack recovery test
    # ============================================================
    attack_name = 'eadl1'
    attack = create_attack(attack_name, wrapped_model, cfg)

    # Compact header
    def_names = list(defenses.keys())
    header = f"{'Mod':<8}{'SNR':>4}{'Clean':>7}{'Att':>7}"
    for dn in def_names:
        header += f"{dn:>8}"
    print("\n" + "=" * len(header))
    print(f"  EADL1 ATTACK RECOVERY — All Defenses")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for snr in [0, 18]:
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
                wrapped_model=wrapped_model, ta_box='minmax',
            )
            att_acc = compute_accuracy(model, x_adv, y)

            row = f"{mod:<8}{snr:>4}{clean_acc*100:>6.1f}%{att_acc*100:>6.1f}%"
            for name, defense_fn in defenses.items():
                if name == 'AddNoise':
                    # Special: returns predictions, not filtered signal
                    preds = additive_noise_defense(x_adv, model)
                    acc = (preds == y).float().mean().item()
                else:
                    x_def = defense_fn(x_adv)
                    acc = compute_accuracy(model, x_def, y)
                row += f"{acc*100:>7.1f}%"
            print(row)
        print()

    # ============================================================
    # Also test CW
    # ============================================================
    attack_cw = create_attack('cw', wrapped_model, cfg)
    print("=" * len(header))
    print(f"  CW ATTACK RECOVERY — All Defenses")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for snr in [0, 18]:
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
                attack_cw, x, y,
                wrapped_model=wrapped_model, ta_box='minmax',
            )
            att_acc = compute_accuracy(model, x_adv, y)

            row = f"{mod:<8}{snr:>4}{clean_acc*100:>6.1f}%{att_acc*100:>6.1f}%"
            for name, defense_fn in defenses.items():
                x_def = defense_fn(x_adv)
                acc = compute_accuracy(model, x_def, y)
                row += f"{acc*100:>7.1f}%"
            print(row)
        print()


if __name__ == '__main__':
    main()
