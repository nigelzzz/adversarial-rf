#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Dataset Generator + AWN Finetuning.

Generates synthetic IQ data via make_rml_like_burst() with configurable channel
presets, then finetunes the pretrained AWN model.  Analog mods (WBFM, AM-DSB,
AM-SSB) are taken from real RML2016 data since our TX chain cannot synthesise them.

Curriculum mode (--curriculum) trains in three progressive stages:
  stage 1 - phase_gain   (phase + gain jitter only)
  stage 2 - phase_gain_cfo (adds CFO)
  stage 3 - full          (adds multipath)

Usage:
    # Generate dataset only
    python synth_finetune.py --mode gen --n_per_cell 1000

    # Single-stage finetune (full preset)
    python synth_finetune.py --mode finetune --n_per_cell 2000 --ft_epochs 50

    # Curriculum finetune (recommended)
    python synth_finetune.py --mode finetune --n_per_cell 2000 --curriculum

    # Evaluate a checkpoint
    python synth_finetune.py --mode eval --ckpt_path ./checkpoint/2016.10a_AWN_ft.pkl
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm

from util.synth_txrx import (
    make_rml_like_burst, PRESETS as CHAN_PRESETS,
    CONSTELLATION_MODS, FSK_MODS,
)
from util.utils import create_model, fix_seed
from util.config import Config
from util.evaluation import Run_Eval
from data_loader.data_loader import Load_Dataset, Dataset_Split, Create_Data_Loader

# Digital mods that our synth chain supports (8 of 11)
SYNTH_MODS = sorted(CONSTELLATION_MODS | FSK_MODS)

# Analog mods only available from real RML2016 data
ANALOG_MODS = ['WBFM', 'AM-DSB', 'AM-SSB']

# AWN class index -> mod name (must match data_loader.py)
IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}

# RML2016.10a SNR range
RML_SNRS = list(range(-20, 20, 2))  # -20 to 18 dB, step 2

# Curriculum stages
CURRICULUM_STAGES = ['phase_gain', 'phase_gain_cfo', 'rml_like']


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_synth_dataset(n_per_cell, snr_list=None, mod_list=None,
                           channel_preset='full', seed=42, target_rms=0.006):
    """Generate synthetic IQ dataset for digital modulations.

    Uses make_rml_like_burst() with the specified channel preset so that
    ablation / curriculum experiments share the same generation path.

    Returns:
        signals: [N, 2, 128] float32
        labels:  [N] int64 (AWN class indices)
        snrs:    [N] int64
    """
    if snr_list is None:
        snr_list = RML_SNRS
    if mod_list is None:
        mod_list = SYNTH_MODS

    rng = np.random.default_rng(seed)
    all_signals, all_labels, all_snrs = [], [], []

    total = len(mod_list) * len(snr_list) * n_per_cell
    with tqdm(total=total, desc=f'  Gen [{channel_preset}]', leave=False) as pbar:
        for mod in mod_list:
            if mod not in MOD_TO_IDX:
                print(f"  Skipping {mod}: not in AWN class map")
                continue
            label_idx = MOD_TO_IDX[mod]
            for snr_db in snr_list:
                cell = []
                for _ in range(n_per_cell):
                    b = make_rml_like_burst(
                        mod, snr_db, channel_preset,
                        target_rms=target_rms, rng=rng,
                    )
                    cell.append(b['iq_tensor'][0])   # [2, 128]
                    pbar.update(1)
                all_signals.append(np.stack(cell, axis=0))
                all_labels.extend([label_idx] * n_per_cell)
                all_snrs.extend([int(snr_db)] * n_per_cell)

    signals = np.concatenate(all_signals, axis=0).astype(np.float32)
    labels  = np.array(all_labels, dtype=np.int64)
    snrs    = np.array(all_snrs,   dtype=np.int64)
    return signals, labels, snrs


def load_analog_from_rml(rml_path='./data/RML2016.10a_dict.pkl', seed=42):
    """Extract WBFM / AM-DSB / AM-SSB samples from real RML2016.

    Returns:
        signals: [N, 2, 128] float32
        labels:  [N] int64
        snrs:    [N] int64
    """
    with open(rml_path, 'rb') as f:
        rml = pickle.load(f, encoding='bytes')

    all_sigs, all_labs, all_snrs = [], [], []
    for mod_name in ANALOG_MODS:
        mod_b = mod_name.encode()
        label_idx = MOD_TO_IDX[mod_name]
        for snr in RML_SNRS:
            key = (mod_b, snr)
            if key not in rml:
                continue
            arr = rml[key].astype(np.float32)   # [N, 2, 128]
            all_sigs.append(arr)
            all_labs.extend([label_idx] * len(arr))
            all_snrs.extend([snr] * len(arr))

    return (np.concatenate(all_sigs, axis=0),
            np.array(all_labs, dtype=np.int64),
            np.array(all_snrs, dtype=np.int64))


def load_rml_train_split(seed=42):
    """Return the same train split used by original AWN training (all 11 mods).

    Returns:
        signals: [N, 2, 128] float32
        labels:  [N] int64
        snrs:    [N] int64 (per-sample SNR)
    """
    import logging
    logger = logging.getLogger('sf_rml_load')
    logger.setLevel(logging.WARNING)
    Signals, Labels, SNRs, snrs, mods = Load_Dataset('2016.10a', logger)
    fix_seed(seed)
    train_set, _, _, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
    Sig_train, Lab_train = train_set
    snrs_all = np.array(SNRs, dtype=np.int64)          # per-sample SNR for all N samples
    # Reconstruct train_idx: Dataset_Split doesn't return it, but we can derive
    # from test_idx + val_idx.  Simplest: re-run same random split to get train_idx.
    # Since we already have Sig_train and Lab_train we just pass dummy snrs.
    return (Sig_train.numpy().astype(np.float32),
            Lab_train.numpy(),
            np.zeros(len(Lab_train), dtype=np.int64))  # snrs not needed for training


def save_synth_dataset(signals, labels, snrs, path):
    """Save synthetic dataset as pickle (same key format as RML2016)."""
    dataset = {}
    for label_idx in np.unique(labels):
        mod_name = IDX_TO_MOD[label_idx]
        mod_key  = mod_name.encode()
        for snr_val in np.unique(snrs):
            mask = (labels == label_idx) & (snrs == snr_val)
            if np.any(mask):
                dataset[(mod_key, int(snr_val))] = signals[mask]
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved synthetic dataset: {path}")
    print(f"  {len(dataset)} cells, {len(labels)} total samples")
    return dataset


def load_synth_dataset(path):
    """Load synthetic dataset from pickle."""
    with open(path, 'rb') as f:
        dataset = pickle.load(f, encoding='bytes')
    sigs, labs, snrs = [], [], []
    classes = {v.encode(): k for k, v in IDX_TO_MOD.items()}
    for (mod_key, snr_val), arr in sorted(dataset.items()):
        label_idx = classes.get(mod_key)
        if label_idx is not None:
            sigs.append(arr)
            labs.extend([label_idx] * len(arr))
            snrs.extend([snr_val] * len(arr))
    return (np.concatenate(sigs, axis=0),
            np.array(labs, dtype=np.int64),
            np.array(snrs, dtype=np.int64))


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(signals, labels, batch_size=256, val_ratio=0.15, seed=42):
    """Split array into train/val DataLoaders."""
    rng = np.random.default_rng(seed)
    n   = len(labels)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _loader(i, shuffle):
        ds = Data.TensorDataset(
            torch.from_numpy(signals[i].astype(np.float32)),
            torch.from_numpy(labels[i]),
        )
        return Data.DataLoader(ds, batch_size=batch_size,
                               shuffle=shuffle, num_workers=2, pin_memory=True)

    print(f"  Dataset: {len(train_idx)} train, {n_val} val")
    return _loader(train_idx, True), _loader(val_idx, False)


# ─────────────────────────────────────────────────────────────────────────────
# Training / finetuning
# ─────────────────────────────────────────────────────────────────────────────

def finetune(model, train_loader, val_loader, device,
             lr=1e-4, epochs=50, patience=8, save_path=None,
             tag='FT'):
    """Finetune AWN model; returns best validation accuracy."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience // 2, verbose=False)

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(epochs):
        # Train
        model.train()
        n_correct = n_total = 0
        for sig, lab in train_loader:
            sig, lab = sig.to(device), lab.to(device)
            logit, regu = model(sig)
            loss = criterion(logit, lab) + sum(regu)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_correct += (logit.argmax(1) == lab).sum().item()
            n_total   += len(lab)
        train_acc = n_correct / n_total

        # Validate
        model.eval()
        v_correct = v_total = v_loss = 0
        with torch.no_grad():
            for sig, lab in val_loader:
                sig, lab = sig.to(device), lab.to(device)
                logit, regu = model(sig)
                v_loss   += (criterion(logit, lab) + sum(regu)).item() * len(lab)
                v_correct += (logit.argmax(1) == lab).sum().item()
                v_total   += len(lab)
        val_acc  = v_correct / v_total
        val_loss = v_loss / v_total
        scheduler.step(val_acc)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"  [{tag}] Ep {epoch+1:3d}: "
              f"train={100*train_acc:.1f}%  val={100*val_acc:.1f}%  "
              f"lr={lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
            if save_path:
                torch.save(best_state, save_path)
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(best val={100*best_val_acc:.2f}%)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path:
            print(f"  Saved: {save_path}  (val={100*best_val_acc:.2f}%)")
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eval_loader(model, loader, device):
    model.eval()
    correct = total = 0
    preds_all, labs_all = [], []
    with torch.no_grad():
        for sig, lab in loader:
            sig, lab = sig.to(device), lab.to(device)
            logit, _ = model(sig)
            preds = logit.argmax(1)
            correct += (preds == lab).sum().item()
            total   += len(lab)
            preds_all.append(preds.cpu())
            labs_all.append(lab.cpu())
    return correct / total, torch.cat(preds_all).numpy(), torch.cat(labs_all).numpy()


def evaluate_on_real(model, device, dataset='2016.10a', seed=42):
    """Evaluate on real RML2016 test set (same split as original training)."""
    import logging
    logger = logging.getLogger('sf_eval'); logger.setLevel(logging.WARNING)
    Signals, Labels, SNRs, snrs, mods = Load_Dataset(dataset, logger)
    cfg         = Config(dataset, train=False)
    cfg.device  = device
    fix_seed(seed)
    _, test_set, _, test_idx = Dataset_Split(Signals, Labels, snrs, mods, logger)
    Sig_test, Lab_test = test_set
    snrs_arr = np.array(SNRs)[test_idx]

    loader = Data.DataLoader(
        Data.TensorDataset(Sig_test, Lab_test),
        batch_size=256, shuffle=False, num_workers=2)
    acc, preds, labs = _eval_loader(model, loader, device)
    print(f"\n  RML2016 test accuracy : {100*acc:.2f}%")
    for snr in range(-20, 20, 4):
        mask = snrs_arr == snr
        if mask.any():
            a = (preds[mask] == labs[mask]).mean()
            print(f"    SNR={snr:3d}: {100*a:.1f}%")
    return acc


def evaluate_on_synth(model, device, signals, labels, snrs_arr=None, tag='Synth'):
    """Evaluate model on a numpy signal array."""
    loader = Data.DataLoader(
        Data.TensorDataset(
            torch.from_numpy(signals.astype(np.float32)),
            torch.from_numpy(labels)),
        batch_size=256, shuffle=False, num_workers=2)
    acc, preds, labs = _eval_loader(model, loader, device)
    print(f"\n  {tag} accuracy : {100*acc:.2f}%")
    if snrs_arr is not None:
        for snr in range(-20, 20, 4):
            mask = snrs_arr == snr
            if mask.any():
                a = (preds[mask] == labs[mask]).mean()
                print(f"    SNR={snr:3d}: {100*a:.1f}%")
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Synthetic Dataset Generator + AWN Finetuning')
    parser.add_argument('--mode', type=str, default='finetune',
                        choices=['gen', 'finetune', 'eval'])
    parser.add_argument('--channel_preset', type=str, default='full',
                        choices=list(CHAN_PRESETS.keys()),
                        help='Channel impairment preset for synthetic data')
    parser.add_argument('--curriculum', action='store_true',
                        help='3-stage curriculum: phase_gain -> phase_gain_cfo -> full')
    parser.add_argument('--n_per_cell', type=int, default=2000,
                        help='Synthetic samples per (mod, snr) cell')
    parser.add_argument('--snr_list', type=str, default=None,
                        help='Comma-separated SNR values (default: full RML range)')
    parser.add_argument('--mod_list', type=str, default=None,
                        help='Comma-separated digital mods (default: all 8)')
    parser.add_argument('--target_rms', type=float, default=0.006,
                        help='Target RMS for synthetic bursts')
    parser.add_argument('--synth_path', type=str,
                        default='./data/synth_2016.10a.pkl',
                        help='Path to save/load synthetic dataset')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint',
                        help='Checkpoint dir (or .pkl path for eval)')
    parser.add_argument('--ft_epochs', type=int, default=50,
                        help='Max finetuning epochs per stage')
    parser.add_argument('--ft_lr', type=float, default=1e-4,
                        help='Finetuning learning rate')
    parser.add_argument('--ft_patience', type=int, default=8,
                        help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--resume', action='store_true',
                        help='Load 2016.10a_AWN_ft.pkl instead of base checkpoint')

    args   = parser.parse_args()
    fix_seed(args.seed)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")

    snr_list = [int(s.strip()) for s in args.snr_list.split(',')] \
        if args.snr_list else None
    mod_list = [m.strip() for m in args.mod_list.split(',')] \
        if args.mod_list else None

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Generate synthetic dataset
    # ──────────────────────────────────────────────────────────────────────────
    if args.mode in ('gen', 'finetune'):
        print("=" * 60)
        print("  Step 1: Generate Synthetic Dataset")
        print("=" * 60)
        preset = args.channel_preset if not args.curriculum else 'full'
        mods   = mod_list or SYNTH_MODS
        snrs   = snr_list or RML_SNRS
        print(f"  Preset : {preset}")
        print(f"  Mods   : {', '.join(mods)}")
        print(f"  SNRs   : {snrs}")
        print(f"  N/cell : {args.n_per_cell}")

        t0 = time.time()
        signals, labels, snrs_arr = generate_synth_dataset(
            n_per_cell=args.n_per_cell,
            snr_list=snrs,
            mod_list=mods,
            channel_preset=preset,
            seed=args.seed,
            target_rms=args.target_rms,
        )
        print(f"\n  Generated {len(labels)} samples in {time.time()-t0:.1f}s")
        print(f"  Shape: {signals.shape}")
        save_synth_dataset(signals, labels, snrs_arr, args.synth_path)

    if args.mode == 'gen':
        return

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Load model + data, finetune
    # ──────────────────────────────────────────────────────────────────────────
    if args.mode == 'finetune':
        print("\n" + "=" * 60)
        print("  Step 2: Finetune AWN")
        print("=" * 60)

        # Load pretrained model
        cfg       = Config('2016.10a', train=False)
        cfg.device = device
        model      = create_model(cfg, model_name='awn')
        ft_ckpt    = os.path.join(args.ckpt_path, '2016.10a_AWN_ft.pkl')
        base_ckpt  = os.path.join(args.ckpt_path, '2016.10a_AWN.pkl')
        ckpt       = ft_ckpt if (args.resume and os.path.exists(ft_ckpt)) else base_ckpt
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device)
        print(f"  Loaded model: {ckpt}")

        # Load full real RML2016 train split (all 11 mods) — prevents forgetting
        print("\n  Loading real RML2016 train split (all 11 mods)...")
        real_sigs, real_labs, _ = load_rml_train_split(seed=args.seed)
        print(f"  Real RML2016 train samples: {len(real_labs)}")

        # Load analog-only slice for the held-out test evaluation
        analog_sigs, analog_labs, analog_snrs = load_analog_from_rml()
        rng = np.random.default_rng(args.seed)
        ap  = rng.permutation(len(analog_labs))
        n_a_test = int(len(analog_labs) * 0.15)
        a_test   = ap[:n_a_test]

        # Synthetic test split (15% held out from the full-preset dataset)
        synth_sigs, synth_labs, synth_snrs = load_synth_dataset(args.synth_path)
        sp       = rng.permutation(len(synth_labs))
        n_s_test = int(len(synth_labs) * 0.15)
        s_test   = sp[:n_s_test]
        s_train  = sp[n_s_test:]

        # Baseline before finetuning
        print("\n  ── Before finetuning ──")
        evaluate_on_real(model, device)
        evaluate_on_synth(model, device,
                          synth_sigs[s_test], synth_labs[s_test],
                          synth_snrs[s_test], tag='Synthetic (full)')

        save_path = os.path.join(args.ckpt_path, '2016.10a_AWN_ft.pkl')

        if args.curriculum:
            # ── Curriculum: 3 stages, each mixes synth + real RML ────────────
            # LR decays each stage; real-RML anchors prevent catastrophic forgetting.
            print("\n  ── Curriculum Finetuning (3 stages, mixed real+synth) ──")
            stage_lrs = [args.ft_lr, args.ft_lr * 0.5, args.ft_lr * 0.25]
            for stage_idx, (stage_preset, stage_lr) in enumerate(
                    zip(CURRICULUM_STAGES, stage_lrs)):
                print(f"\n  Stage {stage_idx+1}/3 — preset={stage_preset}  lr={stage_lr:.2e}")
                stage_sigs, stage_labs, _ = generate_synth_dataset(
                    n_per_cell=args.n_per_cell,
                    snr_list=snr_list or RML_SNRS,
                    mod_list=mod_list or SYNTH_MODS,
                    channel_preset=stage_preset,
                    seed=args.seed + stage_idx,
                    target_rms=args.target_rms,
                )
                # Combine synthetic digital mods + full real RML (anchor)
                all_sigs = np.concatenate([stage_sigs, real_sigs], axis=0)
                all_labs = np.concatenate([stage_labs, real_labs], axis=0)
                print(f"  Mix: {len(stage_sigs)} synth + {len(real_sigs)} real = {len(all_sigs)}")
                train_loader, val_loader = make_loaders(
                    all_sigs, all_labs,
                    batch_size=args.batch_size, seed=args.seed + stage_idx)

                finetune(model, train_loader, val_loader, device,
                         lr=stage_lr,
                         epochs=args.ft_epochs,
                         patience=args.ft_patience,
                         save_path=save_path,
                         tag=f'S{stage_idx+1}:{stage_preset}')
        else:
            # ── Single-stage: full-preset synth + full real RML ───────────────
            all_sigs = np.concatenate([synth_sigs[s_train], real_sigs], axis=0)
            all_labs = np.concatenate([synth_labs[s_train], real_labs], axis=0)
            print(f"\n  Training on {len(all_labs)} samples "
                  f"({len(s_train)} synth + {len(real_sigs)} real)")
            train_loader, val_loader = make_loaders(
                all_sigs, all_labs,
                batch_size=args.batch_size, seed=args.seed)
            finetune(model, train_loader, val_loader, device,
                     lr=args.ft_lr,
                     epochs=args.ft_epochs,
                     patience=args.ft_patience,
                     save_path=save_path,
                     tag='FT')

        # ── Final evaluation ──────────────────────────────────────────────────
        print("\n  ── After finetuning ──")
        evaluate_on_real(model, device)
        evaluate_on_synth(model, device,
                          synth_sigs[s_test], synth_labs[s_test],
                          synth_snrs[s_test], tag='Synthetic (full)')
        evaluate_on_synth(model, device,
                          analog_sigs[a_test], analog_labs[a_test],
                          analog_snrs[a_test], tag='Analog (real)')

    # ──────────────────────────────────────────────────────────────────────────
    # Eval mode
    # ──────────────────────────────────────────────────────────────────────────
    if args.mode == 'eval':
        print("=" * 60)
        print("  Evaluate Model")
        print("=" * 60)

        cfg       = Config('2016.10a', train=False)
        cfg.device = device
        model      = create_model(cfg, model_name='awn')

        ckpt_file = args.ckpt_path
        if os.path.isdir(ckpt_file):
            ft_path   = os.path.join(ckpt_file, '2016.10a_AWN_ft.pkl')
            orig_path = os.path.join(ckpt_file, '2016.10a_AWN.pkl')
            ckpt_file = ft_path if os.path.exists(ft_path) else orig_path

        model.load_state_dict(
            torch.load(ckpt_file, map_location=device, weights_only=True))
        model.to(device)
        print(f"  Loaded: {ckpt_file}")

        evaluate_on_real(model, device)

        if os.path.exists(args.synth_path):
            synth_sigs, synth_labs, synth_snrs = load_synth_dataset(args.synth_path)
            evaluate_on_synth(model, device, synth_sigs, synth_labs,
                              synth_snrs, tag='Synthetic')


if __name__ == '__main__':
    main()
