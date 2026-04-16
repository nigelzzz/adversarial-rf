#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Training Script for AWN on RML2016.10a.

Implements the full AT pipeline (AT-01..AT-05): loads SNR >= 0 dB data,
generates adversarial examples with FGSM/PGD/EADL1/EADEN, and trains the
AWN model with a dual-batch loss (alpha * adv + (1-alpha) * clean).

Analog modulations (WBFM, AM-DSB, AM-SSB) receive clean input in the
adversarial stream to prevent forgetting.

Model warm-starts from the pretrained AWN checkpoint before first optimizer step.

Artifacts produced at runtime:
  ./checkpoint/2016.10a_AWN_at.pkl         - best-epoch state_dict
  ./checkpoint/2016.10a_AWN_at.config.json - training hyperparameters (D-16)
  ./checkpoint/2016.10a_AWN_at_log.csv     - per-epoch metrics (D-17)

Usage:
    # Smoke test (2 epochs)
    python adv_train.py --mode train --epochs 2 --batch_size 64

    # Full training run
    python adv_train.py --mode train --epochs 30 --batch_size 256
"""

import argparse
import csv
import json
import logging
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

import torchattacks

from util.adv_attack import Model01Wrapper, iq_to_ta_input_minmax, ta_output_to_iq_minmax
from data_loader.data_loader import Load_Dataset, Dataset_Split
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

def make_attacks(wrapped_model, eps=0.1, pgd_steps=7, ead_iters=10, ead_bss=1):
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
    # torchattacks EADL1/EADEN compute `iteration % (max_iterations // 10)`
    # internally, which raises ZeroDivisionError when max_iterations < 10.
    ead_iters_safe = max(ead_iters, 10)

    attacks = {
        'fgsm': torchattacks.FGSM(wrapped_model, eps=eps),
        'pgd':  torchattacks.PGD(wrapped_model, eps=eps, alpha=eps / 4,
                                  steps=pgd_steps),
        'eadl1': torchattacks.EADL1(wrapped_model, kappa=0, lr=0.01,
                                     max_iterations=ead_iters_safe,
                                     binary_search_steps=ead_bss),
        'eaden': torchattacks.EADEN(wrapped_model, kappa=0, lr=0.01,
                                     max_iterations=ead_iters_safe,
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
# Training orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def adv_train(model, wrapped_model, train_loader, val_loader, attacks, device, args):
    """
    Full adversarial training loop with ReduceLROnPlateau, best-epoch checkpoint
    selection, per-epoch CSV logging, and early stopping.

    Per D-12, D-13, D-15, D-17.

    Args:
        model:          AWN model (nn.Module)
        wrapped_model:  Model01Wrapper wrapping model
        train_loader:   DataLoader for training data
        val_loader:     DataLoader for validation data
        attacks:        dict of attack name -> torchattacks attack object
        device:         torch.device
        args:           argparse Namespace with lr, alpha, epochs, patience, ckpt_path

    Returns:
        (best_epoch, best_weighted, epochs_trained) tuple
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    # mode='max': we monitor weighted accuracy (higher is better), NOT loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4)

    best_weighted = -1.0
    best_epoch = -1
    no_improve = 0
    ckpt_path = os.path.join(args.ckpt_path, '2016.10a_AWN_at.pkl')
    log_path = os.path.join(args.ckpt_path, '2016.10a_AWN_at_log.csv')

    os.makedirs(args.ckpt_path, exist_ok=True)

    log_fields = ['epoch', 'lr', 'train_loss', 'train_loss_clean', 'train_loss_adv',
                  'val_clean_acc', 'val_robust_fgsm_acc', 'val_weighted', 'time_s']
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()
    log_file.flush()

    epochs_trained = 0
    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss, train_loss_clean, train_loss_adv = train_epoch(
                model, wrapped_model, train_loader, attacks, optimizer,
                criterion, args.alpha, device)
            val_clean_acc, val_robust_fgsm_acc = val_epoch(
                model, wrapped_model, val_loader, attacks, criterion, device)
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']
            epochs_trained = epoch + 1

            # Weighted validation metric per D-12
            val_weighted = 0.5 * val_clean_acc + 0.5 * val_robust_fgsm_acc

            # CSV log row per D-17
            log_writer.writerow({
                'epoch': epoch + 1,
                'lr': f'{lr_now:.2e}',
                'train_loss': f'{train_loss:.6f}',
                'train_loss_clean': f'{train_loss_clean:.6f}',
                'train_loss_adv': f'{train_loss_adv:.6f}',
                'val_clean_acc': f'{val_clean_acc:.4f}',
                'val_robust_fgsm_acc': f'{val_robust_fgsm_acc:.4f}',
                'val_weighted': f'{val_weighted:.4f}',
                'time_s': f'{elapsed:.1f}',
            })
            log_file.flush()

            # Console print
            print(f"Ep {epoch+1:3d}/{args.epochs}: "
                  f"loss={train_loss:.4f} (clean={train_loss_clean:.4f} adv={train_loss_adv:.4f}) "
                  f"val_clean={100*val_clean_acc:.1f}% val_robust={100*val_robust_fgsm_acc:.1f}% "
                  f"weighted={100*val_weighted:.1f}% lr={lr_now:.2e} [{elapsed:.0f}s]")

            # Best-epoch checkpoint per D-12, D-15
            if val_weighted > best_weighted:
                best_weighted = val_weighted
                best_epoch = epoch + 1
                torch.save(model.state_dict(), ckpt_path)
                no_improve = 0
                print(f"  >> New best (weighted={100*val_weighted:.1f}%), saved {ckpt_path}")
            else:
                no_improve += 1

            # Scheduler step per D-13
            scheduler.step(val_weighted)

            # Early stopping per D-13
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (best epoch={best_epoch}, "
                      f"weighted={100*best_weighted:.1f}%)")
                break
    finally:
        log_file.close()

    print(f"\nTraining complete. Best epoch: {best_epoch}, weighted: {100*best_weighted:.1f}%")
    return (best_epoch, best_weighted, epochs_trained)


# ─────────────────────────────────────────────────────────────────────────────
# Config persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_config(path, args, best_epoch, best_weighted, epochs_trained):
    """Save training hyperparameters to JSON. Called after training completes."""
    try:
        sha = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        sha = 'unknown'
    cfg = {
        'epochs_total': args.epochs,
        'epochs_trained': epochs_trained,
        'lr_initial': args.lr,
        'batch_size': args.batch_size,
        'ta_box': 'minmax',
        'eps': args.eps,
        'attack_iters': {
            'fgsm': 1,
            'pgd': args.pgd_steps,
            'eadl1': args.ead_iters,
            'eaden': args.ead_iters,
        },
        'attack_list': list(ATTACK_NAMES),
        'alpha': args.alpha,
        'snr_filter': 'snr_min=0',
        'analog_mods_kept_clean': ['WBFM', 'AM-DSB', 'AM-SSB'],
        'best_epoch': best_epoch,
        'best_val_weighted': round(best_weighted, 6),
        'seed': args.seed,
        'warm_start_ckpt': args.warm_start,
        'git_sha': sha,
    }
    with open(path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Post-training sanity evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_eval(model, device, args):
    """Evaluate AT checkpoint on full RML test set (all SNRs) as sanity check."""
    ckpt_path = os.path.join(args.ckpt_path, '2016.10a_AWN_at.pkl')
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    logger = logging.getLogger('at_sanity')
    logger.setLevel(logging.WARNING)

    # Load full dataset (all SNRs)
    Signals, Labels, SNRs, snrs, mods = Load_Dataset(args.dataset, logger)
    # Use Dataset_Split to get the standard test set
    _, test_set, _, test_idx = Dataset_Split(
        Signals, Labels, snrs, mods, logger)
    Sig_test, Lab_test = test_set

    # Per-SNR accuracy
    print("\n=== Sanity Eval: AT checkpoint on full RML test set ===")
    snrs_arr = np.array([SNRs[i] for i in test_idx])
    preds_all = []
    loader = Data.DataLoader(
        Data.TensorDataset(Sig_test, Lab_test),
        batch_size=256, shuffle=False)
    with torch.no_grad():
        for sig, lab in loader:
            sig = sig.to(device)
            logit, _ = model(sig)
            preds_all.append(logit.argmax(1).cpu())
    preds = torch.cat(preds_all).numpy()
    labs = Lab_test.numpy()

    overall_acc = (preds == labs).mean()
    print(f"Overall test accuracy: {100*overall_acc:.2f}%")
    for snr in sorted(set(snrs_arr)):
        mask = snrs_arr == snr
        if mask.any():
            acc = (preds[mask] == labs[mask]).mean()
            print(f"  SNR={snr:3d} dB: {100*acc:.1f}%")

    # Per-class accuracy for analog classes (AT-02 success criterion)
    print("\nAnalog class accuracy (catastrophic forgetting check):")
    for cls_idx, cls_name in IDX_TO_MOD.items():
        if cls_idx in ANALOG_INDICES:
            cls_mask = labs == cls_idx
            if cls_mask.any():
                cls_acc = (preds[cls_mask] == labs[cls_mask]).mean()
                print(f"  {cls_name} (idx={cls_idx}): {100*cls_acc:.1f}%")


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
    parser.add_argument('--ead_iters', type=int, default=10,
                        help='EAD max_iterations (min 10; default: 10, per D-03)')
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
    print(f"  alpha={args.alpha}, lr={args.lr}, patience={args.patience}")
    print()

    # ── Training with graceful interrupt handling ────────────────────────────
    best_epoch = -1
    best_weighted = -1.0
    epochs_trained = 0
    try:
        best_epoch, best_weighted, epochs_trained = adv_train(
            model, wrapped, train_loader, val_loader, attacks, device, args)
    except KeyboardInterrupt:
        print("\nInterrupted. Saving config for partial run...")
    finally:
        config_path = os.path.join(args.ckpt_path, '2016.10a_AWN_at.config.json')
        save_config(config_path, args, best_epoch, best_weighted, epochs_trained)
        if best_epoch > 0:
            run_sanity_eval(model, device, args)


if __name__ == '__main__':
    main()
