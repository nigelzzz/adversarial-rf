#!/usr/bin/env python
"""
Synthetic Dataset Generator + AWN Finetuning.

Generates synthetic IQ data using the same TX/RX chain as the CRC experiment,
with channel impairments (multipath fading, SRO) to match RML2016.10a
characteristics. Then finetunes AWN on mixed real+synthetic data.

Steps:
  1. Generate synthetic [N, 2, 128] IQ data for all digital mods × SNRs
  2. (Optional) Save as pickle for reuse
  3. Finetune AWN on mixed RML2016 + synthetic data
  4. Evaluate on both real and synthetic test sets

Usage:
    # Generate dataset only
    python synth_finetune.py --mode gen --n_per_cell 1000

    # Generate + finetune
    python synth_finetune.py --mode finetune --n_per_cell 1000 --ft_epochs 20

    # Evaluate finetuned model
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
    generate_burst, random_multipath_taps,
    CONSTELLATION_MODS, FSK_MODS,
)
from util.utils import create_model, fix_seed
from util.config import Config
from util.evaluation import Run_Eval
from data_loader.data_loader import Load_Dataset, Dataset_Split, Create_Data_Loader

# Digital mods that our synth chain supports (8 of 11)
SYNTH_MODS = sorted(CONSTELLATION_MODS | FSK_MODS)

# AWN class index -> mod name (must match data_loader.py)
IDX_TO_MOD = {
    0: 'QAM16', 1: 'QAM64', 2: '8PSK', 3: 'WBFM', 4: 'BPSK',
    5: 'CPFSK', 6: 'AM-DSB', 7: 'GFSK', 8: 'PAM4', 9: 'QPSK', 10: 'AM-SSB',
}
MOD_TO_IDX = {v: k for k, v in IDX_TO_MOD.items()}

# RML2016.10a SNR range
RML_SNRS = list(range(-20, 20, 2))  # -20 to 18 dB, step 2


def compute_psd_templates(rml_path='./data/RML2016.10a_dict.pkl'):
    """Compute per-modulation average PSD from RML2016 (SNR>=10 for clean shape)."""
    with open(rml_path, 'rb') as f:
        rml = pickle.load(f, encoding='bytes')

    templates = {}
    for mod_name, label_idx in MOD_TO_IDX.items():
        mod_b = mod_name.encode()
        psds = []
        for snr in [10, 12, 14, 16, 18]:
            key = (mod_b, snr)
            if key not in rml:
                continue
            arr = rml[key]
            iq = arr[:, 0, :] + 1j * arr[:, 1, :]
            psds.append(np.mean(np.abs(np.fft.fft(iq, axis=1)) ** 2, axis=0))
        if psds:
            templates[mod_name] = np.mean(psds, axis=0)
    return templates


def spectral_shape(signals_2d, psd_target, psd_synth):
    """Apply spectral shaping to match target PSD profile.

    Args:
        signals_2d: [N, 2, 128] array
        psd_target: [128] target PSD from RML2016
        psd_synth: [128] average PSD of synth signals

    Returns:
        shaped [N, 2, 128] array with same RMS as input
    """
    from scipy.ndimage import uniform_filter1d
    H = np.sqrt(psd_target / (psd_synth + 1e-20))
    H = uniform_filter1d(H, size=5, mode='wrap')

    out = np.empty_like(signals_2d)
    for i in range(len(signals_2d)):
        iq = signals_2d[i, 0, :] + 1j * signals_2d[i, 1, :]
        rms_before = np.sqrt(np.mean(np.abs(iq) ** 2))
        S = np.fft.fft(iq)
        iq_shaped = np.fft.ifft(S * H)
        rms_after = np.sqrt(np.mean(np.abs(iq_shaped) ** 2))
        if rms_after > 0:
            iq_shaped = iq_shaped * (rms_before / rms_after)
        out[i, 0, :] = iq_shaped.real.astype(np.float32)
        out[i, 1, :] = iq_shaped.imag.astype(np.float32)
    return out


def generate_synth_dataset(n_per_cell, snr_list=None, mod_list=None,
                           multipath=True, sro=True, seed=42,
                           target_rms=0.00854, spectral_match=True,
                           cfo_std=0.005):
    """Generate synthetic IQ dataset matching RML2016.10a format.

    Args:
        n_per_cell: samples per (mod, snr) combination
        snr_list: list of SNR values (default: RML2016 range)
        mod_list: list of modulation types (default: all 8 digital mods)
        multipath: enable random multipath fading
        sro: enable random sample rate offset
        seed: random seed
        spectral_match: apply spectral shaping to match RML2016 PSD profile
        cfo_std: carrier frequency offset std (default 0.005, ~4 rad over 128 samples)

    Returns:
        signals: [N, 2, 128] numpy array
        labels: [N] int array (AWN class indices)
        snrs: [N] int array
    """
    if snr_list is None:
        snr_list = RML_SNRS
    if mod_list is None:
        mod_list = SYNTH_MODS

    rng = np.random.default_rng(seed)

    # Load PSD templates for spectral shaping
    psd_templates = None
    if spectral_match:
        try:
            psd_templates = compute_psd_templates()
            print("  Spectral shaping: enabled (matching RML2016 PSD)")
        except FileNotFoundError:
            print("  Spectral shaping: disabled (RML2016 data not found)")

    all_signals = []
    all_labels = []
    all_snrs = []

    for mod in mod_list:
        if mod not in MOD_TO_IDX:
            print(f"  Skipping {mod}: not in AWN class map")
            continue
        label_idx = MOD_TO_IDX[mod]

        # Collect all cells for this mod (for spectral shaping)
        mod_signals = []
        mod_snr_counts = []

        for snr_db in snr_list:
            signals_cell = []
            for _ in range(n_per_cell):
                # Random channel impairments
                mp_taps = None
                if multipath:
                    mp_taps = random_multipath_taps(
                        rng, n_taps=5, max_delay_spread=1.0)

                sro_ppm = 0.0
                if sro:
                    sro_ppm = rng.normal(0, 5.0)  # ±5 PPM typical

                b = generate_burst(
                    mod, n_symbols=16, n_pilots=2, sps=8, beta=0.35,
                    snr_db=snr_db, target_rms=target_rms,
                    cfo_std=cfo_std, rng=rng, n_guard=16,
                    multipath_taps=mp_taps, sro_ppm=sro_ppm,
                )
                signals_cell.append(b['iq_tensor'][0])  # [2, 128]

            cell_arr = np.stack(signals_cell, axis=0)  # [n_per_cell, 2, 128]
            mod_signals.append(cell_arr)
            mod_snr_counts.append((snr_db, n_per_cell))

        # Stack all cells for this mod
        mod_arr = np.concatenate(mod_signals, axis=0)

        # Apply spectral shaping per modulation
        if psd_templates is not None and mod in psd_templates:
            psd_target = psd_templates[mod]
            # Compute synth average PSD for this mod
            iq_all = mod_arr[:, 0, :] + 1j * mod_arr[:, 1, :]
            psd_synth = np.mean(np.abs(np.fft.fft(iq_all, axis=1)) ** 2, axis=0)
            mod_arr = spectral_shape(mod_arr, psd_target, psd_synth)

        # Split back into cells and append
        offset = 0
        for snr_db, count in mod_snr_counts:
            all_signals.append(mod_arr[offset:offset + count])
            all_labels.extend([label_idx] * count)
            all_snrs.extend([snr_db] * count)
            offset += count

    signals = np.concatenate(all_signals, axis=0)  # [N, 2, 128]
    labels = np.array(all_labels, dtype=np.int64)
    snrs = np.array(all_snrs, dtype=np.int64)

    return signals, labels, snrs


def save_synth_dataset(signals, labels, snrs, path):
    """Save synthetic dataset as pickle (same key format as RML2016)."""
    # Build dict keyed by (mod_bytes, snr_int) like RML2016
    dataset = {}
    for label_idx in np.unique(labels):
        mod_name = IDX_TO_MOD[label_idx]
        mod_key = mod_name.encode()
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

    signals_list = []
    labels_list = []
    snrs_list = []

    classes = {v.encode(): k for k, v in IDX_TO_MOD.items()}

    for (mod_key, snr_val), arr in sorted(dataset.items()):
        signals_list.append(arr)
        label_idx = classes.get(mod_key)
        if label_idx is not None:
            labels_list.extend([label_idx] * len(arr))
            snrs_list.extend([snr_val] * len(arr))

    signals = np.concatenate(signals_list, axis=0)
    labels = np.array(labels_list, dtype=np.int64)
    snrs = np.array(snrs_list, dtype=np.int64)
    return signals, labels, snrs


def make_mixed_loaders(real_signals, real_labels, synth_signals, synth_labels,
                       batch_size=128, val_ratio=0.15, seed=42):
    """Create train/val DataLoaders from mixed real + synthetic data."""
    rng = np.random.default_rng(seed)

    # Concatenate
    all_signals = np.concatenate([real_signals, synth_signals], axis=0)
    all_labels = np.concatenate([real_labels, synth_labels], axis=0)

    n = len(all_labels)
    indices = rng.permutation(n)
    n_val = int(n * val_ratio)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_ds = Data.TensorDataset(
        torch.from_numpy(all_signals[train_idx].astype(np.float32)),
        torch.from_numpy(all_labels[train_idx]),
    )
    val_ds = Data.TensorDataset(
        torch.from_numpy(all_signals[val_idx].astype(np.float32)),
        torch.from_numpy(all_labels[val_idx]),
    )

    train_loader = Data.DataLoader(train_ds, batch_size=batch_size,
                                   shuffle=True, num_workers=2)
    val_loader = Data.DataLoader(val_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=2)

    print(f"Mixed dataset: {len(train_ds)} train, {len(val_ds)} val")
    print(f"  Real: {len(real_labels)}, Synth: {len(synth_labels)}")
    return train_loader, val_loader


def finetune(model, train_loader, val_loader, device,
             lr=0.0001, epochs=20, patience=5, save_path=None,
             freeze_conv=False):
    """Finetune AWN model with lower learning rate.

    Args:
        freeze_conv: If True, freeze conv1/conv2 layers (only train wavelet +
                     classifier). Helps preserve learned low-level features.
    """
    model.to(device)

    if freeze_conv:
        for name, param in model.named_parameters():
            if name.startswith('conv1') or name.startswith('conv2'):
                param.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Frozen conv1/conv2. Trainable params: {sum(p.numel() for p in trainable)}")
        optimizer = torch.optim.Adam(trainable, lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for sig_batch, lab_batch in tqdm(train_loader,
                                         desc=f'FT Epoch {epoch+1}/{epochs}',
                                         leave=False):
            sig_batch = sig_batch.to(device)
            lab_batch = lab_batch.to(device)

            logit, regu_sum = model(sig_batch)
            loss = criterion(logit, lab_batch) + sum(regu_sum)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(lab_batch)
            train_correct += (logit.argmax(1) == lab_batch).sum().item()
            train_total += len(lab_batch)

        train_acc = train_correct / train_total
        train_loss = train_loss_sum / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for sig_batch, lab_batch in val_loader:
                sig_batch = sig_batch.to(device)
                lab_batch = lab_batch.to(device)
                logit, regu_sum = model(sig_batch)
                loss = criterion(logit, lab_batch) + sum(regu_sum)
                val_loss_sum += loss.item() * len(lab_batch)
                val_correct += (logit.argmax(1) == lab_batch).sum().item()
                val_total += len(lab_batch)

        val_acc = val_correct / val_total
        val_loss = val_loss_sum / val_total
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:3d}: train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f} val_loss={val_loss:.4f} lr={lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path:
            torch.save(best_state, save_path)
            print(f"  Saved best model: {save_path} (val_acc={best_val_acc:.4f})")

    return best_val_acc


def evaluate_on_real(model, device, dataset='2016.10a'):
    """Evaluate model on real RML2016 test set."""
    import logging
    logger = logging.getLogger('synth_ft_eval')
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    Signals, Labels, SNRs, snrs, mods = Load_Dataset(dataset, logger)
    cfg = Config(dataset, train=False)
    cfg.device = device

    # Use same split as original training
    fix_seed(42)
    train_set, test_set, val_set, test_idx = Dataset_Split(
        Signals, Labels, snrs, mods, logger)
    Signals_test, Labels_test = test_set

    test_ds = Data.TensorDataset(Signals_test, Labels_test)
    test_loader = Data.DataLoader(test_ds, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    # Per-SNR accuracy
    snr_correct = {}
    snr_total = {}

    with torch.no_grad():
        for sig_batch, lab_batch in test_loader:
            sig_batch = sig_batch.to(device)
            lab_batch = lab_batch.to(device)
            logit, _ = model(sig_batch)
            preds = logit.argmax(1)
            correct += (preds == lab_batch).sum().item()
            total += len(lab_batch)

    overall_acc = correct / total
    print(f"\n  Real RML2016 test accuracy: {100*overall_acc:.2f}% ({correct}/{total})")
    return overall_acc


def evaluate_on_synth(model, device, synth_signals, synth_labels):
    """Evaluate model on synthetic test data."""
    ds = Data.TensorDataset(
        torch.from_numpy(synth_signals.astype(np.float32)),
        torch.from_numpy(synth_labels),
    )
    loader = Data.DataLoader(ds, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sig_batch, lab_batch in loader:
            sig_batch = sig_batch.to(device)
            lab_batch = lab_batch.to(device)
            logit, _ = model(sig_batch)
            preds = logit.argmax(1)
            correct += (preds == lab_batch).sum().item()
            total += len(lab_batch)

    acc = correct / total
    print(f"  Synthetic test accuracy: {100*acc:.2f}% ({correct}/{total})")
    return acc


def main():
    parser = argparse.ArgumentParser(
        description='Synthetic Dataset Generator + AWN Finetuning')
    parser.add_argument('--mode', type=str, default='finetune',
                        choices=['gen', 'finetune', 'eval'],
                        help='gen: generate only, finetune: gen+finetune, eval: evaluate')
    parser.add_argument('--n_per_cell', type=int, default=500,
                        help='Synthetic samples per (mod, snr) cell')
    parser.add_argument('--snr_list', type=str, default=None,
                        help='Comma-separated SNR values (default: full RML range)')
    parser.add_argument('--mod_list', type=str, default=None,
                        help='Comma-separated mods (default: all 8 digital)')
    parser.add_argument('--no_multipath', action='store_true',
                        help='Disable multipath fading')
    parser.add_argument('--no_sro', action='store_true',
                        help='Disable sample rate offset')
    parser.add_argument('--no_spectral_match', action='store_true',
                        help='Disable spectral shaping to match RML2016 PSD')
    parser.add_argument('--target_rms', type=float, default=0.00854,
                        help='Target RMS amplitude (default: 0.00854, matching RML2016 complex RMS)')
    parser.add_argument('--synth_path', type=str,
                        default='./data/synth_2016.10a.pkl',
                        help='Path to save/load synthetic dataset')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoint',
                        help='Model checkpoint directory (or path for eval)')
    parser.add_argument('--ft_epochs', type=int, default=20,
                        help='Finetuning epochs')
    parser.add_argument('--ft_lr', type=float, default=0.0001,
                        help='Finetuning learning rate')
    parser.add_argument('--ft_patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--cfo_std', type=float, default=0.005,
                        help='CFO std (default 0.005; RML2016 generation used ~0.01)')
    parser.add_argument('--freeze_conv', action='store_true',
                        help='Freeze conv1/conv2 layers during finetuning')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    fix_seed(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    snr_list = None
    if args.snr_list:
        snr_list = [int(s.strip()) for s in args.snr_list.split(',')]

    mod_list = None
    if args.mod_list:
        mod_list = [m.strip() for m in args.mod_list.split(',')]

    # =====================================================
    # Step 1: Generate synthetic dataset
    # =====================================================
    if args.mode in ('gen', 'finetune'):
        print("=" * 60)
        print("  Step 1: Generate Synthetic Dataset")
        print("=" * 60)

        mods_str = ', '.join(mod_list or SYNTH_MODS)
        snrs_str = str(snr_list or RML_SNRS)
        print(f"  Mods: {mods_str}")
        print(f"  SNRs: {snrs_str}")
        print(f"  Samples/cell: {args.n_per_cell}")
        print(f"  Multipath: {not args.no_multipath}")
        print(f"  SRO: {not args.no_sro}")
        print(f"  Target RMS: {args.target_rms}")

        t0 = time.time()
        signals, labels, snrs = generate_synth_dataset(
            n_per_cell=args.n_per_cell,
            snr_list=snr_list,
            mod_list=mod_list,
            multipath=not args.no_multipath,
            sro=not args.no_sro,
            seed=args.seed,
            target_rms=args.target_rms,
            spectral_match=not args.no_spectral_match,
            cfo_std=args.cfo_std,
        )
        elapsed = time.time() - t0
        print(f"\n  Generated {len(labels)} samples in {elapsed:.1f}s")
        print(f"  Shape: {signals.shape}")
        print(f"  RMS: {np.sqrt(np.mean(signals**2)):.6f}")

        # Per-mod counts
        for idx in sorted(np.unique(labels)):
            mod_name = IDX_TO_MOD[idx]
            count = np.sum(labels == idx)
            print(f"    {mod_name}: {count}")

        save_synth_dataset(signals, labels, snrs, args.synth_path)

    if args.mode == 'gen':
        print("\nDone (generation only).")
        return

    # =====================================================
    # Step 2: Load real + synthetic data, create loaders
    # =====================================================
    if args.mode == 'finetune':
        print("\n" + "=" * 60)
        print("  Step 2: Finetune AWN on Mixed Data")
        print("=" * 60)

        # Load real RML2016 data
        import logging
        logger = logging.getLogger('synth_ft')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        Signals_real, Labels_real, SNRs_real, snrs_r, mods_r = Load_Dataset(
            '2016.10a', logger)
        real_signals = Signals_real.numpy()
        real_labels = Labels_real.numpy()

        # Load synthetic
        synth_signals, synth_labels, synth_snrs = load_synth_dataset(
            args.synth_path)

        # Split synthetic into train (85%) / test (15%)
        rng = np.random.default_rng(args.seed)
        n_synth = len(synth_labels)
        synth_perm = rng.permutation(n_synth)
        n_synth_test = int(n_synth * 0.15)
        synth_test_idx = synth_perm[:n_synth_test]
        synth_train_idx = synth_perm[n_synth_test:]

        synth_test_signals = synth_signals[synth_test_idx]
        synth_test_labels = synth_labels[synth_test_idx]

        # Mixed train+val from real + synthetic train portion
        train_loader, val_loader = make_mixed_loaders(
            real_signals, real_labels,
            synth_signals[synth_train_idx], synth_labels[synth_train_idx],
            batch_size=args.batch_size, seed=args.seed,
        )

        # Load pretrained model
        cfg = Config('2016.10a', train=False)
        cfg.device = device
        model = create_model(cfg, model_name='awn')

        ckpt_file = os.path.join(args.ckpt_path, '2016.10a_AWN.pkl')
        state_dict = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Loaded pretrained model: {ckpt_file}")

        # Evaluate before finetuning
        print("\n  --- Before finetuning ---")
        model.to(device)
        evaluate_on_real(model, device)
        evaluate_on_synth(model, device, synth_test_signals, synth_test_labels)

        # Finetune
        print("\n  --- Finetuning ---")
        save_path = os.path.join(args.ckpt_path, '2016.10a_AWN_ft.pkl')
        best_acc = finetune(
            model, train_loader, val_loader, device,
            lr=args.ft_lr, epochs=args.ft_epochs,
            patience=args.ft_patience, save_path=save_path,
            freeze_conv=args.freeze_conv,
        )

        # Evaluate after finetuning
        print("\n  --- After finetuning ---")
        evaluate_on_real(model, device)
        evaluate_on_synth(model, device, synth_test_signals, synth_test_labels)

    # =====================================================
    # Eval mode: just evaluate a checkpoint
    # =====================================================
    if args.mode == 'eval':
        print("=" * 60)
        print("  Evaluate Model")
        print("=" * 60)

        cfg = Config('2016.10a', train=False)
        cfg.device = device
        model = create_model(cfg, model_name='awn')

        ckpt_file = args.ckpt_path
        if os.path.isdir(ckpt_file):
            # Try finetuned first, fall back to original
            ft_path = os.path.join(ckpt_file, '2016.10a_AWN_ft.pkl')
            orig_path = os.path.join(ckpt_file, '2016.10a_AWN.pkl')
            ckpt_file = ft_path if os.path.exists(ft_path) else orig_path

        state_dict = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"  Loaded: {ckpt_file}")

        evaluate_on_real(model, device)

        if os.path.exists(args.synth_path):
            synth_signals, synth_labels, _ = load_synth_dataset(args.synth_path)
            evaluate_on_synth(model, device, synth_signals, synth_labels)


if __name__ == '__main__':
    main()
