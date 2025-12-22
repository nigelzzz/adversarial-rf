import os
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from util.detector import RFSignalAutoEncoder, normalize_for_detector, kl_divergence_timewise


def _make_loaders(x_train: torch.Tensor,
                  x_val: torch.Tensor,
                  batch_size: int,
                  num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(x_train)
    val_ds = TensorDataset(x_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train_detector(
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    *,
    device: torch.device,
    out_path: str,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 5,
    num_workers: int = 0,
    logger=None,
) -> str:
    """
    Train the RFSignalAutoEncoder to reconstruct normalized inputs.
    Saves best checkpoint to out_path and returns the path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model = RFSignalAutoEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    best_val = float('inf')
    bad = 0

    train_loader, val_loader = _make_loaders(x_train, x_val, batch_size, num_workers)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            xb_n = normalize_for_detector(xb)
            recon = model(xb_n)
            loss = crit(recon, xb_n)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
            n_tr += xb.size(0)
        tr_loss /= max(1, n_tr)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                xb_n = normalize_for_detector(xb)
                recon = model(xb_n)
                loss = crit(recon, xb_n)
                va_loss += float(loss.item()) * xb.size(0)
                n_va += xb.size(0)
        va_loss /= max(1, n_va)

        if logger:
            logger.info(f"[AE] epoch {ep}/{epochs} | train_mse={tr_loss:.6f} val_mse={va_loss:.6f}")
        else:
            print(f"[AE] epoch {ep}/{epochs} | train_mse={tr_loss:.6f} val_mse={va_loss:.6f}")

        if va_loss + 1e-12 < best_val:
            best_val = va_loss
            bad = 0
            torch.save(model.state_dict(), out_path)
        else:
            bad += 1
            if bad >= patience:
                if logger:
                    logger.info("[AE] early stopping")
                break

    return out_path


@torch.no_grad()
def calibrate_threshold(
    x_val: torch.Tensor,
    ckpt_path: str,
    *,
    device: torch.device,
    quantile: float = 0.90,
    batch_size: int = 512,
    num_workers: int = 0,
    logger=None,
) -> float:
    """
    Load a trained AE and compute KL divergences on clean validation data,
    then return the quantile threshold (e.g., 0.90 ⇒ 90th percentile).
    """
    model = RFSignalAutoEncoder().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(TensorDataset(x_val), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    kls = []
    for (xb,) in loader:
        xb = xb.to(device)
        xb_n = normalize_for_detector(xb)
        recon = model(xb_n)
        kl = kl_divergence_timewise(xb_n, recon)
        kls.append(kl.cpu())
    kl_all = torch.cat(kls, dim=0)
    thr = float(torch.quantile(kl_all, q=quantile))
    if logger:
        logger.info(f"[AE] KL quantile={quantile:.2f} threshold={thr:.6f}")
    else:
        print(f"[AE] KL quantile={quantile:.2f} threshold={thr:.6f}")
    return thr

