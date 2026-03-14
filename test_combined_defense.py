#!/usr/bin/env python
"""
Combined defense: chain two fast signal-processing steps, ONE model inference.
No need to know modulation. No multiple inferences.

Key idea: combine defenses that are good at different things.
- Quant: destroys LSB perturbations, saves AM-SSB
- LP/Top-K: removes high-freq adversarial components
- Chain them: signal processing only, <0.5ms total
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
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min + 1e-12)
    x_quant = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
    return x_quant * (x_max - x_min) + x_min

def median_filter(x, kernel_size=5):
    N, C, T = x.shape
    pad = kernel_size // 2
    x_padded = F_torch.pad(x, (pad, pad), mode='reflect')
    windows = x_padded.unfold(dimension=2, size=kernel_size, step=1)
    return windows.median(dim=3).values

def moving_avg_smooth(x, kernel_size=3):
    N, C, T = x.shape
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size
    x_flat = x.view(N * C, 1, T)
    x_smooth = F_torch.conv1d(x_flat, kernel, padding=pad)
    return x_smooth.view(N, C, T)

def soft_lowpass(x, cutoff_bin=20, rolloff=5):
    """Lowpass with smooth rolloff (raised cosine taper)."""
    N, C, T = x.shape
    X = torch.fft.rfft(x, n=T, dim=2)
    n_bins = X.shape[2]
    mask = torch.ones(n_bins, device=x.device)
    for i in range(rolloff):
        idx = cutoff_bin + i
        if idx < n_bins:
            mask[idx] = 0.5 * (1 + np.cos(np.pi * (i + 1) / (rolloff + 1)))
    mask[cutoff_bin + rolloff:] = 0.0
    return torch.fft.irfft(X * mask.view(1, 1, -1), n=T, dim=2)


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

    # Combined defenses: all are pure signal processing, no model inference
    defenses = {
        'Top-20':      lambda x: fft_topk_denoise(x, topk=20),
        'LP-20':       lambda x: lowpass_filter(x, cutoff_bin=20),
        'Quant32':     lambda x: input_quantize(x, n_levels=32),
        'Q64→LP20':    lambda x: lowpass_filter(input_quantize(x, n_levels=64), cutoff_bin=20),
        'Q64→LP25':    lambda x: lowpass_filter(input_quantize(x, n_levels=64), cutoff_bin=25),
        'Q64→LP30':    lambda x: lowpass_filter(input_quantize(x, n_levels=64), cutoff_bin=30),
        'Q32→LP20':    lambda x: lowpass_filter(input_quantize(x, n_levels=32), cutoff_bin=20),
        'Q32→LP25':    lambda x: lowpass_filter(input_quantize(x, n_levels=32), cutoff_bin=25),
        'Q32→LP30':    lambda x: lowpass_filter(input_quantize(x, n_levels=32), cutoff_bin=30),
        'LP20→Q32':    lambda x: input_quantize(lowpass_filter(x, cutoff_bin=20), n_levels=32),
        'LP20→Q64':    lambda x: input_quantize(lowpass_filter(x, cutoff_bin=20), n_levels=64),
        'Q64→T20':     lambda x: fft_topk_denoise(input_quantize(x, n_levels=64), topk=20),
        'Q32→T20':     lambda x: fft_topk_denoise(input_quantize(x, n_levels=32), topk=20),
        'Q64→SLP25':   lambda x: soft_lowpass(input_quantize(x, n_levels=64), cutoff_bin=25, rolloff=5),
        'MvA3→Q64':    lambda x: input_quantize(moving_avg_smooth(x, kernel_size=3), n_levels=64),
    }

    attacks_to_test = ['eadl1', 'cw']

    for attack_name in attacks_to_test:
        attack = create_attack(attack_name, wrapped_model, cfg)

        for snr in [0, 18]:
            print(f"\n{'='*160}")
            print(f"  {attack_name.upper()} | SNR={snr}")
            print(f"{'='*160}")
            hdr = f"{'Mod':<8}{'Clean':>7}{'Att':>7}"
            for dn in defenses:
                hdr += f"{dn:>10}"
            print(hdr)
            print("-" * 160)

            all_accs = {dn: [] for dn in defenses}
            all_accs['clean'] = []
            all_accs['att'] = []

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

                row = f"{mod:<8}{clean_acc*100:>6.1f}%{att_acc*100:>6.1f}%"
                all_accs['clean'].append(clean_acc)
                all_accs['att'].append(att_acc)

                for dn, dfn in defenses.items():
                    x_def = dfn(x_adv)
                    acc = compute_accuracy(model, x_def, y)
                    row += f"{acc*100:>9.1f}%"
                    all_accs[dn].append(acc)

                print(row)

            print("-" * 160)
            row = f"{'AVG':<8}{'':>7}{'':>7}"
            for dn in defenses:
                avg = np.mean(all_accs[dn])
                row += f"{avg*100:>9.1f}%"
            print(row)
            print()


if __name__ == '__main__':
    main()
