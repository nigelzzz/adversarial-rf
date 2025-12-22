import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RFSignalAutoEncoder(nn.Module):
    """
    Lightweight 1D conv autoencoder used as an anomaly detector on IQ signals.
    Architecture adapted from AWN_All.py.
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.enc_bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.3)
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2)
        self.enc_bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        self.enc_conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=4, dilation=4)
        self.enc_bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)

        # Channel attention on latent
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.Sigmoid(),
        )

        # Decoder
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(64)
        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(32)
        self.dec_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn3 = nn.BatchNorm1d(16)
        self.dec_conv4 = nn.ConvTranspose1d(16, 2, kernel_size=3, stride=1, padding=1)

        # Skip path to improve reconstruction
        self.skip1 = nn.Conv1d(2, 128, kernel_size=1, stride=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.dropout1(F.relu(self.enc_bn1(self.enc_conv1(x))))
        x2 = self.dropout2(F.relu(self.enc_bn2(self.enc_conv2(x1))))
        x3 = self.dropout3(F.relu(self.enc_bn3(self.enc_conv3(x2))))
        x4 = self.dropout4(F.relu(self.enc_bn4(self.enc_conv4(x3))))

        # Attention
        attn = self.attention(x4)
        x4 = x4 * attn

        # Decode with skip
        x4 = x4 + self.skip1(x)
        x5 = F.relu(self.dec_bn1(self.dec_conv1(x4)))
        x6 = F.relu(self.dec_bn2(self.dec_conv2(x5)))
        x7 = F.relu(self.dec_bn3(self.dec_conv3(x6)))
        x8 = self.dec_conv4(x7)
        return x8


def normalize_for_detector(x: torch.Tensor, offset: float = 0.02, scale: float = 0.04) -> torch.Tensor:
    """Simple affine normalization used by the detector pipeline."""
    return (x + float(offset)) / float(scale)


@torch.no_grad()
def kl_divergence_timewise(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute a KL-like divergence per sample by softmax-normalizing over time
    for each channel and summing across channels and time. Matches AWN_All.py logic.

    Args:
        p, q: tensors of shape [N, 2, T]
    Returns:
        kl: shape [N] with per-sample divergence
    """
    assert p.shape == q.shape and p.dim() == 3 and p.size(1) == 2
    p_s = F.softmax(p, dim=-1)
    q_s = F.softmax(q, dim=-1)
    kl = (p_s * (p_s.clamp_min(1e-12).log() - q_s.clamp_min(1e-12).log())).sum(dim=(1, 2))
    return kl


@torch.no_grad()
def detector_gate_fft_topk(
    x: torch.Tensor,
    detector: RFSignalAutoEncoder,
    *,
    threshold: float,
    topk: int,
    norm_offset: float = 0.02,
    norm_scale: float = 0.04,
    apply_in_normalized: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply AE-based gating: only inputs with KL divergence above `threshold`
    are denoised using FFT top-k, others pass through.

    Returns the processed tensor and the per-sample KL values.
    """
    x_norm = normalize_for_detector(x, offset=norm_offset, scale=norm_scale)
    recon = detector(x_norm)
    kl = kl_divergence_timewise(x_norm, recon)

    # Build mask of samples to denoise
    to_denoise = kl > float(threshold)
    if to_denoise.any():
        from util.defense import fft_topk_denoise
        x_proc = x.clone()
        if apply_in_normalized:
            # Apply Top-K in normalized domain and map back to original scale
            xs = x[to_denoise]
            xs_n = normalize_for_detector(xs, offset=norm_offset, scale=norm_scale)
            xs_n_filt = fft_topk_denoise(xs_n, topk=topk)
            xs_back = xs_n_filt * float(norm_scale) - float(norm_offset)
            x_proc[to_denoise] = xs_back
        else:
            x_proc[to_denoise] = fft_topk_denoise(x[to_denoise], topk=topk)
    else:
        x_proc = x
    return x_proc, kl
