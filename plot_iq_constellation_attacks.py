#!/usr/bin/env python
"""
Plot IQ constellation: clean vs adversarial with paper normalization.

Phase alignment (M-th power de-rotation) is estimated from the CLEAN signal
and applied identically to the adversarial signal, preserving perturbation
displacement. Paper normalization (x + 0.02) / 0.04 maps to [0, 1] axis.

Usage:
    python plot_iq_constellation_attacks.py --snr 10 --modulations BPSK --attacks fgsm
    python plot_iq_constellation_attacks.py --snr 10 --modulations BPSK,QPSK,QAM16,QAM64
    python plot_iq_constellation_attacks.py --snr 10 --modulations BPSK --attacks fgsm --eps 0.3
"""

import os
import pickle
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchattacks

# ── Dataset / model constants ───────────────────────────────────────────
DATASET = '2016.10a'
DATASET_PATH = './data/RML2016.10a_dict.pkl'
CKPT_PATH = './checkpoint'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = {
    b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
    b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9,
    b'AM-SSB': 10,
}
CLASS_NAMES = [
    'QAM16', 'QAM64', '8PSK', 'WBFM', 'BPSK',
    'CPFSK', 'AM-DSB', 'GFSK', 'PAM4', 'QPSK', 'AM-SSB',
]

# M-th power phase estimation order per modulation
PHASE_ORDER = {
    'BPSK': 2, 'QPSK': 4, '8PSK': 8, 'QAM16': 4, 'QAM64': 4,
    'PAM4': 2, 'WBFM': 1, 'AM-DSB': 1, 'AM-SSB': 1, 'CPFSK': 4, 'GFSK': 4,
}


# ── Model helpers ───────────────────────────────────────────────────────

def create_model():
    from models.model import AWN
    import yaml

    cfg = yaml.safe_load(open(f'./config/{DATASET}.yml', 'r'))
    model = AWN(
        num_classes=cfg['num_classes'],
        num_levels=cfg.get('num_levels', cfg.get('num_level', 1)),
        kernel_size=cfg['kernel_size'],
        in_channels=cfg['in_channels'],
        latent_dim=cfg['latent_dim'],
    )
    ckpt_file = os.path.join(CKPT_PATH, f'{DATASET}_AWN.pkl')
    model.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def get_attack(attack_name, model_wrapper, eps=0.03):
    name = attack_name.lower()
    if name == 'fgsm':
        return torchattacks.FGSM(model_wrapper, eps=eps)
    elif name == 'pgd':
        return torchattacks.PGD(model_wrapper, eps=eps, alpha=eps / 4, steps=10)
    elif name == 'bim':
        return torchattacks.BIM(model_wrapper, eps=eps, alpha=eps / 4, steps=10)
    elif name == 'cw':
        return torchattacks.CW(model_wrapper, c=1.0, kappa=0, steps=50, lr=0.01)
    elif name == 'deepfool':
        return torchattacks.DeepFool(model_wrapper, steps=50)
    elif name == 'apgd':
        return torchattacks.APGD(model_wrapper, eps=eps, steps=10)
    elif name == 'mifgsm':
        return torchattacks.MIFGSM(model_wrapper, eps=eps, alpha=eps / 4, steps=10)
    elif name == 'rfgsm':
        return torchattacks.RFGSM(model_wrapper, eps=eps, alpha=eps / 2, steps=10)
    elif name == 'upgd':
        return torchattacks.UPGD(model_wrapper, eps=eps, alpha=eps / 4, steps=10)
    elif name == 'eotpgd':
        return torchattacks.EOTPGD(model_wrapper, eps=eps, alpha=eps / 4, steps=10)
    elif name == 'vmifgsm':
        return torchattacks.VMIFGSM(model_wrapper, eps=eps, alpha=eps / 4, steps=10)
    elif name == 'vnifgsm':
        return torchattacks.VNIFGSM(model_wrapper, eps=eps, alpha=eps / 4, steps=10)
    elif name == 'jitter':
        return torchattacks.Jitter(model_wrapper, eps=eps, alpha=eps / 4, steps=10)
    elif name == 'ffgsm':
        return torchattacks.FFGSM(model_wrapper, eps=eps, alpha=eps)
    elif name == 'pgdl2':
        return torchattacks.PGDL2(model_wrapper, eps=eps * 10, alpha=eps * 2, steps=10)
    elif name == 'eadl1':
        return torchattacks.EADL1(model_wrapper, kappa=0, lr=0.01, max_iterations=50)
    elif name == 'eaden':
        return torchattacks.EADEN(model_wrapper, kappa=0, lr=0.01, max_iterations=50)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def run_attack(model_wrapper, signals, labels, attack_name, eps, ta_box):
    from util.adv_attack import (iq_to_ta_input, ta_output_to_iq,
                                  iq_to_ta_input_minmax, ta_output_to_iq_minmax,
                                  iq_to_ta_input_paper, ta_output_to_iq_paper)

    attack = get_attack(attack_name, model_wrapper, eps=eps)

    if ta_box == 'paper':
        model_wrapper.set_minmax(
            a=torch.tensor(-0.02, device=signals.device),
            b=torch.tensor(0.04, device=signals.device),
        )
        x01_4d = iq_to_ta_input_paper(signals)
        adv01_4d = attack(x01_4d, labels)
        adv_iq = ta_output_to_iq_paper(adv01_4d)
        model_wrapper.clear_minmax()
    elif ta_box == 'minmax':
        x01_4d, a, b = iq_to_ta_input_minmax(signals)
        model_wrapper.set_minmax(a, b)
        adv01_4d = attack(x01_4d, labels)
        adv_iq = ta_output_to_iq_minmax(adv01_4d, a, b)
        model_wrapper.clear_minmax()
    else:
        x01_4d = iq_to_ta_input(signals)
        adv01_4d = attack(x01_4d, labels)
        adv_iq = ta_output_to_iq(adv01_4d)

    return adv_iq


# ── Phase alignment ────────────────────────────────────────────────────

def phase_align_clean(clean_np, mod_name):
    """
    Per-sample M-th power phase de-rotation on clean signals (no RMS normalization).
    Amplitude stays in raw IQ range so paper normalization works afterwards.

    Returns:
        aligned: np.ndarray [N, 2, L] — de-rotated clean IQ
        phases:  list of float — estimated phase per sample (to reuse on adversarial)
    """
    N = clean_np.shape[0]
    order = PHASE_ORDER.get(mod_name, 4)
    aligned = np.empty_like(clean_np)
    phases = []

    for k in range(N):
        iq = clean_np[k, 0, :] + 1j * clean_np[k, 1, :]

        if order > 1:
            phase_est = np.angle(np.mean(iq ** order)) / order
        else:
            phase_est = 0.0
        iq = iq * np.exp(-1j * phase_est)

        aligned[k, 0, :] = iq.real
        aligned[k, 1, :] = iq.imag
        phases.append(phase_est)

    return aligned, phases


def apply_clean_phase(adv_np, phases):
    """
    Apply the CLEAN signal's phase de-rotation to adversarial signals.
    Preserves perturbation displacement in the constellation.
    """
    N = adv_np.shape[0]
    aligned = np.empty_like(adv_np)

    for k in range(N):
        iq = adv_np[k, 0, :] + 1j * adv_np[k, 1, :]
        iq = iq * np.exp(-1j * phases[k])
        aligned[k, 0, :] = iq.real
        aligned[k, 1, :] = iq.imag

    return aligned


# ── Normalization for plotting ─────────────────────────────────────────

def paper_normalize(x_np):
    """Paper normalization: (x + 0.02) / 0.04, maps [-0.02, 0.02] → [0, 1]."""
    return (x_np + 0.02) / 0.04


def collect_scatter_points(normed_np, max_points=15000):
    """Flatten [N, 2, L] → (I, Q) arrays and subsample."""
    I_all = normed_np[:, 0, :].ravel()
    Q_all = normed_np[:, 1, :].ravel()

    if len(I_all) > max_points:
        sel = np.random.choice(len(I_all), max_points, replace=False)
        I_all = I_all[sel]
        Q_all = Q_all[sel]

    return I_all, Q_all


# ── Plotting ────────────────────────────────────────────────────────────

def plot_clean_vs_attack(clean_I, clean_Q, adv_I, adv_Q, mod_name,
                         attack_name, snr, ax_clean, ax_adv, axis_lim=1.0):
    """Draw side-by-side scatter on pre-created axes."""
    ax_clean.scatter(clean_I, clean_Q, s=2, alpha=0.35, c='#1f77b4',
                     edgecolors='none', rasterized=True)
    ax_clean.set_title(f'{mod_name} — Clean', fontsize=11)
    ax_clean.set_xlim(0, axis_lim)
    ax_clean.set_ylim(0, axis_lim)
    ax_clean.set_aspect('equal')
    ax_clean.grid(True, alpha=0.2, linewidth=0.5)
    ax_clean.tick_params(labelsize=8)

    ax_adv.scatter(adv_I, adv_Q, s=2, alpha=0.35, c='#d62728',
                   edgecolors='none', rasterized=True)
    ax_adv.set_title(f'{mod_name} — {attack_name.upper()}', fontsize=11)
    ax_adv.set_xlim(0, axis_lim)
    ax_adv.set_ylim(0, axis_lim)
    ax_adv.set_aspect('equal')
    ax_adv.grid(True, alpha=0.2, linewidth=0.5)
    ax_adv.tick_params(labelsize=8)


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='IQ constellation plots: clean vs adversarial '
                    '(paper normalization)')
    p.add_argument('--snr', type=int, default=10)
    p.add_argument('--samples', type=int, default=200,
                   help='Samples per modulation to use')
    p.add_argument('--modulations', type=str, default='BPSK,QPSK,QAM16,QAM64',
                   help='Comma-separated modulation names')
    p.add_argument('--attacks', type=str, default='fgsm',
                   help='Comma-separated attack names')
    p.add_argument('--eps', type=float, default=0.3,
                   help='Epsilon for Linf attacks')
    p.add_argument('--ta_box', type=str, default='paper',
                   choices=['unit', 'minmax', 'paper'],
                   help='Normalization mode for torchattacks')
    p.add_argument('--output_dir', type=str, default='./iq_constellation_attacks')
    p.add_argument('--points', type=int, default=15000,
                   help='Max scatter points per subplot')
    p.add_argument('--axis_lim', type=float, default=1.0,
                   help='Axis upper limit (paper-normalized [0, 1])')
    return p.parse_args()


def main():
    from util.adv_attack import Model01Wrapper

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(42)

    modulations = [m.strip() for m in args.modulations.split(',')]
    attacks = [a.strip().lower() for a in args.attacks.split(',')]

    # ── Load dataset ────────────────────────────────────────────────────
    print(f'Loading {DATASET_PATH} ...')
    with open(DATASET_PATH, 'rb') as f:
        Xd = pickle.load(f, encoding='bytes')

    # ── Load model ──────────────────────────────────────────────────────
    print('Loading model ...')
    model = create_model()
    model_wrapper = Model01Wrapper(model).eval()

    # ── Process each (modulation, attack) pair ──────────────────────────
    for attack_name in attacks:
        n_mods = len(modulations)
        fig, axes = plt.subplots(n_mods, 2, figsize=(6, 3 * n_mods),
                                 squeeze=False)
        fig.suptitle(
            f'{attack_name.upper()} (eps={args.eps}, {args.ta_box})  '
            f'SNR={args.snr} dB',
            fontsize=13, fontweight='bold', y=1.01)

        for row, mod_name in enumerate(modulations):
            mod_bytes = mod_name.encode()
            key = (mod_bytes, args.snr)

            if key not in Xd:
                print(f'  WARNING: {mod_name} @ SNR={args.snr} not found')
                continue

            # Raw clean IQ: [N, 2, 128]
            clean_np = Xd[key][:args.samples].copy().astype(np.float32)
            N = clean_np.shape[0]
            print(f'  {mod_name}: {N} samples')

            # 1) Generate adversarial examples
            clean_t = torch.from_numpy(clean_np).to(DEVICE)
            class_idx = CLASSES[mod_bytes]
            labels_t = torch.full((N,), class_idx, dtype=torch.long,
                                  device=DEVICE)

            try:
                with torch.enable_grad():
                    adv_t = run_attack(model_wrapper, clean_t, labels_t,
                                       attack_name, eps=args.eps,
                                       ta_box=args.ta_box)
                adv_np = adv_t.cpu().detach().numpy()
            except Exception as e:
                print(f'    ERROR running {attack_name}: {e}')
                adv_np = clean_np.copy()

            # 2) Phase-align (de-rotate) using clean reference, then
            #    paper-normalize for plotting
            clean_derot, phases = phase_align_clean(clean_np, mod_name)
            adv_derot = apply_clean_phase(adv_np, phases)
            clean_normed = paper_normalize(clean_derot)
            adv_normed = paper_normalize(adv_derot)

            # 3) Collect scatter points
            cI, cQ = collect_scatter_points(clean_normed,
                                            max_points=args.points)
            aI, aQ = collect_scatter_points(adv_normed,
                                            max_points=args.points)

            # 4) Plot
            plot_clean_vs_attack(cI, cQ, aI, aQ, mod_name, attack_name,
                                 args.snr, axes[row, 0], axes[row, 1],
                                 axis_lim=args.axis_lim)

            # Axis labels on edges only
            if row == n_mods - 1:
                axes[row, 0].set_xlabel('In-phase (I)', fontsize=9)
                axes[row, 1].set_xlabel('In-phase (I)', fontsize=9)
            axes[row, 0].set_ylabel('Quadrature (Q)', fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(
            args.output_dir,
            f'{"_".join(modulations)}_{attack_name}_snr{args.snr}'
            f'_constellation.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out_path}')

    print(f'\nDone. Output in {args.output_dir}/')


if __name__ == '__main__':
    main()
