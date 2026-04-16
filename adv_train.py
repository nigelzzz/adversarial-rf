#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Training Script for AWN on RML2016.10a.

Implements the AT pipeline (AT-01, AT-03, AT-04): loads SNR >= 0 dB data,
generates adversarial examples with FGSM/PGD/EADL1/EADEN, and trains the
AWN model with a dual-batch loss (alpha * adv + (1-alpha) * clean).

Analog modulations (WBFM, AM-DSB, AM-SSB) receive clean input in the
adversarial stream to prevent forgetting.

Model warm-starts from the pretrained AWN checkpoint before first optimizer step.

Plan 02 adds: checkpoint selection, CSV logging, JSON config, scheduler,
early stopping, and post-training eval hook.

Usage:
    # Smoke test (1 epoch)
    python adv_train.py --mode train --epochs 1 --batch_size 64

    # Full training run (Plan 02 adds early stopping)
    python adv_train.py --mode train --epochs 30 --batch_size 256
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

import torchattacks

from util.adv_attack import Model01Wrapper, iq_to_ta_input_minmax, ta_output_to_iq_minmax
from data_loader.data_loader import Load_Dataset
from util.utils import create_model, fix_seed
from util.config import Config

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# AWN class index -> mod name (matches data_loader.py and synth_finetune.py)
IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}

# Analog class indices: WBFM=3, AM-DSB=6, AM-SSB=10
# These mods cannot be synthesized; always use clean input in adversarial stream.
ANALOG_INDICES = {3, 6, 10}

# Attacks used in AT training (CW is held out for evaluation)
ATTACK_NAMES = ['fgsm', 'pgd', 'eadl1', 'eaden']


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(dataset='2016.10a', batch_size=256, val_ratio=0.15,
                  seed=42, snr_min=0):
    """
    Load RML dataset filtered to SNR >= snr_min and split 85/15 train/val.

    Args:
        dataset:    Dataset name (e.g. '2016.10a')
        batch_size: Mini-batch size for DataLoader
        val_ratio:  Fraction of data reserved for validation (default 0.15)
        seed:       Random seed for reproducible split
        snr_min:    Minimum SNR in dB; samples below this are excluded

    Returns:
        train_loader: DataLoader for training
        val_loader:   DataLoader for validation
        Signals:      Full [N,2,L] tensor (pre-split, for external use)
        Labels:       Full [N] label tensor (pre-split)
        SNRs:         List of per-sample SNR values (pre-split)
    """
    # Suppress verbose logging from Load_Dataset
    logger = logging.getLogger('at_loader')
    logger.setLevel(logging.WARNING)

    Signals, Labels, SNRs, snrs, mods = Load_Dataset(dataset, logger, snr_min=snr_min)

    n_total = len(Labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_idx = torch.from_numpy(perm[:n_train].copy())
    val_idx = torch.from_numpy(perm[n_train:].copy())

    train_ds = Data.TensorDataset(Signals[train_idx], Labels[train_idx])
    val_ds = Data.TensorDataset(Signals[val_idx], Labels[val_idx])

    train_loader = Data.DataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True, drop_last=False, num_workers=0)
    val_loader = Data.DataLoader(val_ds, batch_size=batch_size,
                                 shuffle=False, drop_last=False, num_workers=0)

    print(f"  AT dataset: {n_train} train, {n_val} val (SNR >= {snr_min} dB)")
    return train_loader, val_loader, Signals, Labels, SNRs


# ─────────────────────────────────────────────────────────────────────────────
# Attack factory
# ─────────────────────────────────────────────────────────────────────────────

def make_attacks(wrapped_model, eps=0.1, pgd_steps=7, ead_iters=7, ead_bss=1):
    """
    Instantiate FGSM, PGD, EADL1, EADEN attack objects.

    Epsilon is interpreted in minmax-normalised [0,1] space. All attacks
    operate on the Model01Wrapper (which reverses minmax normalisation
    internally before forwarding to AWN).

    Args:
        wrapped_model: Model01Wrapper instance
        eps:           Linf / L1 perturbation budget
        pgd_steps:     PGD iterations
        ead_iters:     EAD max_iterations (kept small for training speed)
        ead_bss:       EAD binary_search_steps (1 = fast, 9 = accurate)

    Returns:
        dict mapping attack name -> torchattacks attack object
    """
    attacks = {
        'fgsm': torchattacks.FGSM(wrapped_model, eps=eps),
        'pgd':  torchattacks.PGD(wrapped_model, eps=eps, alpha=eps / 4,
                                  steps=pgd_steps),
        'eadl1': torchattacks.EADL1(wrapped_model, kappa=0, lr=0.01,
                                     max_iterations=ead_iters,
                                     binary_search_steps=ead_bss),
        'eaden': torchattacks.EADEN(wrapped_model, kappa=0, lr=0.01,
                                     max_iterations=ead_iters,
                                     binary_search_steps=ead_bss),
    }
    return attacks


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial batch generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_adv_batch(wrapped_model, x, y, attack):
    """
    Generate adversarial examples for a mini-batch, with analog substitution.

    Per-sample minmax normalisation maps each IQ burst to [0,1] before
    passing to torchattacks; the inverse mapping restores IQ scale.

    Analog modulations (WBFM, AM-DSB, AM-SSB) cannot be reliably perturbed
    in a semantically meaningful way, so their clean version is substituted
    back after attack generation.

    Args:
        wrapped_model: Model01Wrapper (must be on same device as x)
        x:             [N,2,L] clean IQ tensor
        y:             [N] label tensor
        attack:        torchattacks attack object

    Returns:
        x_adv: [N,2,L] adversarial IQ tensor (detached, no grad)
    """
    # 1. Convert IQ -> minmax-normalised 4D tensor and record affine params
    x01_4d, a, b = iq_to_ta_input_minmax(x)

    # 2. Set wrapper normalisation context for this batch
    wrapped_model.set_minmax(a, b)

    # 3. Generate adversarial examples (torchattacks manages gradient context)
    with torch.enable_grad():
        adv01_4d = attack(x01_4d, y)

    # 4. Clear normalisation context
    wrapped_model.clear_minmax()

    # 5. Invert minmax normalisation back to IQ space
    x_adv = ta_output_to_iq_minmax(adv01_4d, a, b)

    # 6. Analog substitution: replace adversarial with clean for analog mods
    analog_mask = torch.tensor(
        [yi.item() in ANALOG_INDICES for yi in y],
        dtype=torch.bool, device=x.device
    )
    if analog_mask.any():
        x_adv[analog_mask] = x[analog_mask]

    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Training and validation
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, wrapped_model, train_loader, attacks, optimizer,
                criterion, alpha, device):
    """
    One epoch of dual-batch adversarial training.

    For each mini-batch:
      1. Select one attack at random (uniform over ATTACK_NAMES)
      2. Generate adversarial batch
      3. Forward both clean and adversarial batches through the model
      4. Compute loss = alpha * L_adv + (1 - alpha) * L_clean + sum(regu_adv)
      5. Backprop and step

    Args:
        model:          AWN model (nn.Module)
        wrapped_model:  Model01Wrapper wrapping model
        train_loader:   DataLoader for training data
        attacks:        dict of attack name -> torchattacks attack object
        optimizer:      torch.optim.Optimizer
        criterion:      nn.CrossEntropyLoss
        alpha:          Weight for adversarial loss (0.0 = clean only)
        device:         torch.device

    Returns:
        (avg_total_loss, avg_clean_loss, avg_adv_loss) tuple of floats
    """
    model.train()

    total_loss = 0.0
    total_loss_clean = 0.0
    total_loss_adv = 0.0
    n_samples = 0

    for sig, lab in train_loader:
        sig = sig.to(device)
        lab = lab.to(device)

        # Select one attack at random for this batch
        attack_name = random.choice(ATTACK_NAMES)
        attack = attacks[attack_name]

        # Generate adversarial batch
        x_adv = generate_adv_batch(wrapped_model, sig, lab, attack)

        # Clean forward pass
        logit_clean, regu_clean = model(sig)

        # Adversarial forward pass
        logit_adv, regu_adv = model(x_adv)

        # Loss: alpha-weighted sum + AWN regularisation from adversarial pass only
        loss_clean = criterion(logit_clean, lab)
        loss_adv = criterion(logit_adv, lab)
        loss = alpha * loss_adv + (1.0 - alpha) * loss_clean + sum(regu_adv)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = len(lab)
        total_loss += loss.item() * n
        total_loss_clean += loss_clean.item() * n
        total_loss_adv += loss_adv.item() * n
        n_samples += n

    return (total_loss / n_samples,
            total_loss_clean / n_samples,
            total_loss_adv / n_samples)


def val_epoch(model, wrapped_model, val_loader, attacks, criterion, device):
    """
    Compute clean accuracy and FGSM robust accuracy on the validation set.

    FGSM is used as the validation attack (fast, representative proxy for
    overall robustness). Full multi-attack evaluation is deferred to Plan 02.

    Args:
        model:          AWN model
        wrapped_model:  Model01Wrapper
        val_loader:     DataLoader for validation data
        attacks:        dict of attack objects (uses 'fgsm')
        criterion:      nn.CrossEntropyLoss (unused here, kept for API symmetry)
        device:         torch.device

    Returns:
        (val_clean_acc, val_robust_fgsm_acc) floats in [0, 1]
    """
    # ── Clean accuracy ────────────────────────────────────────────────────────
    model.eval()
    n_correct_clean = 0
    n_total = 0

    with torch.no_grad():
        for sig, lab in val_loader:
            sig = sig.to(device)
            lab = lab.to(device)
            logit, _ = model(sig)
            preds = logit.argmax(dim=1)
            n_correct_clean += (preds == lab).sum().item()
            n_total += len(lab)

    val_clean_acc = n_correct_clean / n_total if n_total > 0 else 0.0

    # ── FGSM robust accuracy ─────────────────────────────────────────────────
    n_correct_robust = 0
    fgsm_attack = attacks['fgsm']

    for sig, lab in val_loader:
        sig = sig.to(device)
        lab = lab.to(device)

        # Attack generation requires gradients — switch to train mode temporarily
        model.train()
        x_adv = generate_adv_batch(wrapped_model, sig, lab, fgsm_attack)

        # Inference on adversarial batch in eval mode
        model.eval()
        with torch.no_grad():
            logit, _ = model(x_adv)
            preds = logit.argmax(dim=1)
            n_correct_robust += (preds == lab).sum().item()

    val_robust_fgsm_acc = n_correct_robust / n_total if n_total > 0 else 0.0

    return val_clean_acc, val_robust_fgsm_acc


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Adversarial training for AWN (RML2016.10a)'
    )
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                        help='Execution mode')
    parser.add_argument('--dataset', default='2016.10a',
                        help='Dataset name (default: 2016.10a)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', default='auto',
                        help='Device: "auto", "cuda", or "cpu"')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Mini-batch size (default: 256, per D-13)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Max training epochs (default: 30, per D-13)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Adam learning rate (default: 1e-4, per D-13)')
    parser.add_argument('--patience', type=int, default=8,
                        help='Early stopping patience (default: 8, per D-13)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Adversarial loss weight (default: 0.5, per D-07)')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='Linf/L1 epsilon budget in minmax space (default: 0.1, per D-02)')
    parser.add_argument('--pgd_steps', type=int, default=7,
                        help='PGD iteration count (default: 7, per D-03)')
    parser.add_argument('--ead_iters', type=int, default=7,
                        help='EAD max_iterations (default: 7, per D-03)')
    parser.add_argument('--ead_bss', type=int, default=1,
                        help='EAD binary_search_steps (default: 1, fast training mode)')
    parser.add_argument('--ckpt_path', default='./checkpoint',
                        help='Checkpoint directory for I/O (default: ./checkpoint)')
    parser.add_argument('--warm_start', default='./checkpoint/2016.10a_AWN.pkl',
                        help='Pretrained checkpoint for warm-start (default: ./checkpoint/2016.10a_AWN.pkl)')
    args = parser.parse_args()

    # ── Device resolution ────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"  Using device: {device}")

    # ── Seeding ──────────────────────────────────────────────────────────────
    fix_seed(args.seed)

    # ── Data loading ─────────────────────────────────────────────────────────
    train_loader, val_loader, Signals, Labels, SNRs = build_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        val_ratio=0.15,
        seed=args.seed,
        snr_min=0,
    )

    # ── Model construction and warm-start ────────────────────────────────────
    cfg = Config(args.dataset, train=False)
    cfg.device = device
    model = create_model(cfg, 'awn')
    model.load_state_dict(
        torch.load(args.warm_start, map_location=device, weights_only=True)
    )
    model.to(device)
    print(f"  Warm-started from: {args.warm_start}")

    # ── Wrapped model for torchattacks ───────────────────────────────────────
    wrapped = Model01Wrapper(model)

    # ── Attacks ──────────────────────────────────────────────────────────────
    attacks = make_attacks(wrapped, eps=args.eps, pgd_steps=args.pgd_steps,
                           ead_iters=args.ead_iters, ead_bss=args.ead_bss)
    print(f"  Attacks: {list(attacks.keys())} | eps={args.eps}")

    # ── Optimiser and criterion ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # -- training loop (Plan 02 Task 1 completes this with checkpoint saving,
    #    CSV logging, scheduler, and early stopping) --
    print(f"  Starting {args.epochs} epoch(s) of adversarial training ...")
    print(f"  alpha={args.alpha}, lr={args.lr}, patience={args.patience}")
    print()

    for ep in range(args.epochs):
        train_loss, loss_clean, loss_adv = train_epoch(
            model, wrapped, train_loader, attacks, optimizer, criterion,
            args.alpha, device
        )
        val_clean, val_robust = val_epoch(
            model, wrapped, val_loader, attacks, criterion, device
        )
        print(
            f"Ep {ep + 1:3d}: loss={train_loss:.4f} "
            f"(clean={loss_clean:.4f}, adv={loss_adv:.4f}) | "
            f"val_clean={100 * val_clean:.1f}% "
            f"robust={100 * val_robust:.1f}%"
        )


if __name__ == '__main__':
    main()
