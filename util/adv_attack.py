import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import torch.nn as nn

# Note: This CW-L2 implementation is inspired by the widely used formulation
# (e.g., torchattacks' CW). It optimizes a perturbation in input space and
# adds optional low-pass smoothing tailored for IQ data.


def _lowpass_filter(x: torch.Tensor, kernel_size: int = 17) -> torch.Tensor:
    """
    Apply a simple moving-average low-pass filter along the time axis per IQ channel.
    x: Tensor of shape [N, 2, T]
    """
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    # depthwise conv per channel
    kernel = torch.ones(2, 1, kernel_size, device=x.device, dtype=x.dtype) / float(kernel_size)
    x_padded = F.pad(x, (pad, pad), mode="reflect")
    x_filtered = F.conv1d(x_padded, kernel, bias=None, stride=1, padding=0, groups=2)
    return x_filtered


def _batch_clip(x: torch.Tensor, clip_min: torch.Tensor, clip_max: torch.Tensor) -> torch.Tensor:
    # Allow per-batch scalar min/max or broadcastable tensors
    return torch.max(torch.min(x, clip_max), clip_min)


class Model01Wrapper(nn.Module):
    """
    Wrapper to adapt AWN IQ classifier for torchattacks image-style inputs.

    torchattacks assumes inputs are images in [0,1] with 4D shape (N,C,H,W).
    Our signals are IQ waveforms in [-1,1] with shape (N,2,L).

    This wrapper:
    - Accepts 4D [0,1] tensors shaped [N,2,L,1] or [N,2,1,L] (or 3D [N,2,L])
    - Converts them back to [-1,1] IQ as [N,2,L]
    - Forwards through the base model and returns logits (no softmax)
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self._use_minmax: bool = False
        self._a: Optional[torch.Tensor] = None
        self._b: Optional[torch.Tensor] = None

    def set_minmax(self, a: torch.Tensor, b: torch.Tensor):
        """
        Provide per-sample affine mapping parameters so that wrapper interprets
        inputs as x = a + b * x01 instead of the default x = 2*x01 - 1.

        a, b shapes must be broadcastable to [N,2,L]. Typical shape: [N,1,1].
        """
        self._use_minmax = True
        self._a = a
        self._b = b

    def clear_minmax(self):
        self._use_minmax = False
        self._a, self._b = None, None

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        # Normalize shape to [N,2,L]
        if x01.dim() == 4 and x01.shape[-1] == 1:
            x01 = x01.squeeze(-1)
        elif x01.dim() == 4 and x01.shape[-2] == 1:
            x01 = x01.squeeze(-2)
        elif x01.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected input shape for IQ wrapper: {tuple(x01.shape)}")

        if self._use_minmax:
            if self._a is None or self._b is None:
                raise RuntimeError("minmax mapping not set; call set_minmax(a,b) first")
            x_iq = x01 * self._b + self._a
        else:
            x_iq = 2.0 * x01 - 1.0  # [0,1] -> [-1,1]
        logits, _ = self.base(x_iq)
        return logits


def iq_to_ta_input(x_iq: torch.Tensor) -> torch.Tensor:
    """
    Convert IQ batch in [-1,1], shape [N,2,L], to torchattacks-friendly [0,1]
    4D shape [N,2,L,1].
    """
    assert x_iq.dim() == 3 and x_iq.size(1) == 2, "Expected [N,2,L] IQ tensor"
    x01 = (x_iq + 1.0) / 2.0
    return x01.unsqueeze(-1)


def ta_output_to_iq(x01_4d: torch.Tensor) -> torch.Tensor:
    """Inverse of iq_to_ta_input: [N,2,L,1] in [0,1] -> [N,2,L] in [-1,1]."""
    if x01_4d.dim() == 4:
        x01 = x01_4d.squeeze(-1)
    elif x01_4d.dim() == 3:
        x01 = x01_4d
    else:
        raise ValueError("Unexpected tensor rank for torchattacks output")
    return 2.0 * x01 - 1.0


def iq_to_ta_input_minmax(x_iq: torch.Tensor):
    """
    Per-sample min-max mapping to [0,1]. Returns (x01_4d, a, b) where
    x = a + b * x01. Shapes: a,b are [N,1,1].
    """
    assert x_iq.dim() == 3 and x_iq.size(1) == 2, "Expected [N,2,L] IQ tensor"
    a = x_iq.amin(dim=(1, 2), keepdim=True)
    b = (x_iq.amax(dim=(1, 2), keepdim=True) - a).clamp_min(1e-6)
    x01 = (x_iq - a) / b
    return x01.unsqueeze(-1), a, b


def ta_output_to_iq_minmax(x01_4d: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inverse of iq_to_ta_input_minmax: [N,2,L,1] -> [N,2,L] using x = a + b*x01."""
    if x01_4d.dim() == 4:
        x01 = x01_4d.squeeze(-1)
    elif x01_4d.dim() == 3:
        x01 = x01_4d
    else:
        raise ValueError("Unexpected tensor rank for torchattacks output")
    return x01 * b + a


def cw_l2_attack(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    targeted: bool = False,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 100,
    lr: float = 1e-2,
    lowpass: bool = True,
    lowpass_kernel: int = 17,
    clip_min: Optional[torch.Tensor] = None,
    clip_max: Optional[torch.Tensor] = None,
    device: Optional[Union[torch.device, str]] = None,
):
    """
    Basic CW-L2 attack (untargeted by default) for IQ data.

    Args are standard CW knobs plus optional lowpass smoothing of the perturbation.
    Returns adversarial examples of same shape as x: [N, 2, T].
    """
    if device is None:
        device = x.device
    x = x.detach().to(device)
    y = y.detach().to(device)

    # Per-batch clipping range inferred from inputs if not given
    if clip_min is None:
        clip_min = x.amin(dim=(1, 2), keepdim=True)
    if clip_max is None:
        clip_max = x.amax(dim=(1, 2), keepdim=True)

    # Optimize perturbation directly, then clamp. This is a practical CW variant.
    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(steps):
        adv = x + delta
        adv = _batch_clip(adv, clip_min, clip_max)

        logits, _ = model(adv)

        # Gather true and best-other logits
        one_hot = F.one_hot(y, num_classes=logits.size(1)).float()
        real = (one_hot * logits).sum(dim=1)
        other = (logits - 1e4 * one_hot).max(dim=1)[0]

        if targeted:
            f = torch.clamp(other - real + kappa, min=0)
        else:
            f = torch.clamp(real - other + kappa, min=0)

        l2 = (adv - x).pow(2).flatten(1).sum(dim=1)
        loss = l2.sum() + c * f.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lowpass:
            # Keep perturbation low-frequency by filtering delta in-place
            with torch.no_grad():
                delta.data = _lowpass_filter(delta.data, kernel_size=lowpass_kernel)

    adv = _batch_clip(x + delta.detach(), clip_min, clip_max)
    return adv


def spectral_noise_attack(
    x: torch.Tensor,
    *,
    spec_type: str = "cw_tone",
    spec_eps: Optional[float] = 0.1,
    jnr_db: Optional[float] = None,
    tone_freq: Optional[float] = None,
    band: Optional[Tuple[float, float]] = None,
    psd_mask: Optional[torch.Tensor] = None,
    clip_min: Optional[torch.Tensor] = None,
    clip_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Add spectrally-shaped perturbations without running an optimizer.

    Currently supports a CW tone jammer in complex baseband (I/Q).

    Args:
        x: Input batch, shape [N, 2, T] with channels (I, Q).
        spec_type: One of {"cw_tone", "psd_band", "psd_mask"}.
        spec_eps: Target L2 norm of the perturbation per-sample (across I/Q). If None and
                  jnr_db is provided, derives amplitude from JNR.
        jnr_db: Jammer-to-noise ratio in dB (relative to per-sample RMS of x). If provided,
                sets perturbation power accordingly. Ignored if spec_eps is set.
        tone_freq: Normalized digital frequency in cycles per sample in [0, 0.5]. If None,
                   a random frequency is chosen per sample.
        band: (f_low, f_high) normalized in [0, 0.5] for flat band-limited PSD when
              spec_type == "psd_band".
        psd_mask: Optional one-sided PSD mask for rFFT, shape [T//2+1] or [N, T//2+1]
                  when spec_type == "psd_mask".
        clip_min/clip_max: Per-sample clipping bounds. If None, inferred from x.

    Returns:
        x_adv: Perturbed batch with same shape as x.
    """
    assert x.dim() == 3 and x.size(1) == 2, "Expected input of shape [N, 2, T] (I/Q)."
    N, C, T = x.shape

    device = x.device
    dtype = x.dtype

    if clip_min is None:
        clip_min = x.amin(dim=(1, 2), keepdim=True)
    if clip_max is None:
        clip_max = x.amax(dim=(1, 2), keepdim=True)

    # Determine amplitude from either L2 eps or JNR
    if spec_eps is not None:
        # For I/Q tone with I=A*cos and Q=A*sin, L2^2 over T samples is A^2 * T
        A = float(spec_eps) / (T ** 0.5)
    else:
        if jnr_db is None:
            raise ValueError("Provide either spec_eps or jnr_db to set tone amplitude.")
        # Compute per-sample RMS of the input across I/Q/time
        rms = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()  # [N,1,1]
        jnr = 10.0 ** (jnr_db / 20.0)
        # Set tone RMS = rms / jnr, for both I/Q combined; tone RMS equals A/sqrt(2) over I/Q,
        # but since cos^2+sin^2=1 per sample, effective RMS across channels equals A.
        A_batch = (rms / jnr).to(dtype)

    n = torch.arange(T, device=device, dtype=dtype)[None, None, :].repeat(N, 1, 1)  # [N,1,T]

    st = spec_type.lower()
    if st == "cw_tone":
        # Choose frequency per sample if not specified
        if tone_freq is None:
            # Random in (0, 0.5) to avoid DC/Nyquist
            f = torch.rand(N, 1, 1, device=device, dtype=dtype) * 0.5
        else:
            f = torch.full((N, 1, 1), float(tone_freq), device=device, dtype=dtype)
        # Random phase per sample
        phi = torch.rand(N, 1, 1, device=device, dtype=dtype) * (2.0 * torch.pi)

        # Build I/Q tone: I = A*cos(2pi f n + phi), Q = A*sin(...)
        theta = 2.0 * torch.pi * f * n + phi
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        if spec_eps is not None:
            A_batch = torch.full((N, 1, 1), A, device=device, dtype=dtype)

        i_tone = A_batch * cos_t
        q_tone = A_batch * sin_t
        delta = torch.cat([i_tone, q_tone], dim=1)  # [N,2,T]
    elif st in ("psd_band", "band"):
        if band is None:
            raise ValueError("band=(f_low,f_high) must be provided for spec_type='psd_band'")
        f_low, f_high = band
        if not (0.0 <= f_low < f_high <= 0.5):
            raise ValueError("band edges must satisfy 0 <= f_low < f_high <= 0.5")
        # Build one-sided mask for rFFT: length T//2+1
        F = T // 2 + 1
        mask = torch.zeros(F, device=device, dtype=dtype)
        k_low = int(torch.ceil(torch.tensor(f_low * T)).item())
        k_high = int(torch.floor(torch.tensor(f_high * T)).item())
        k_low = max(0, min(k_low, F - 1))
        k_high = max(0, min(k_high, F - 1))
        if k_high <= k_low:
            k_high = min(k_low + 1, F - 1)
        mask[k_low:k_high + 1] = 1.0

        # Sample shaped noise in frequency, irfft to time per channel
        def _sample_time_from_mask(mask_1s: torch.Tensor) -> torch.Tensor:
            F = mask_1s.numel()
            # Random complex spectrum with variance proportional to mask
            real = torch.randn(N, F, device=device, dtype=dtype)
            imag = torch.randn(N, F, device=device, dtype=dtype)
            # DC and Nyquist bins must be real-valued; zero imag there
            imag[:, 0] = 0.0
            if T % 2 == 0:
                imag[:, -1] = 0.0
            X = (real + 1j * imag) * mask_1s[None, :]
            y = torch.fft.irfft(X, n=T, dim=1)
            return y  # [N, T]

        i_noise = _sample_time_from_mask(mask)
        q_noise = _sample_time_from_mask(mask)
        delta = torch.stack([i_noise, q_noise], dim=1)

        # Normalize to desired magnitude (L2 or JNR)
        if spec_eps is not None:
            l2 = delta.pow(2).flatten(1).sum(dim=1).sqrt().clamp_min(1e-12)  # [N]
            scale = (float(spec_eps) / l2).view(N, 1, 1)
            delta = delta * scale
        else:
            if jnr_db is None:
                raise ValueError("Provide either spec_eps or jnr_db to set noise amplitude.")
            rms_x = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
            rms_delta = delta.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
            jnr = 10.0 ** (jnr_db / 20.0)
            target_rms = rms_x / jnr
            delta = delta * (target_rms / rms_delta)

    elif st in ("psd_mask", "mask"):
        if psd_mask is None:
            raise ValueError("psd_mask must be provided for spec_type='psd_mask'")
        # Prepare mask shape [N, F] with F=T//2+1
        F = T // 2 + 1
        if psd_mask.dim() == 1:
            if psd_mask.numel() != F:
                raise ValueError(f"psd_mask length must be {F} for T={T}")
            mask = psd_mask.to(device=device, dtype=dtype).unsqueeze(0).repeat(N, 1)
        elif psd_mask.dim() == 2:
            if psd_mask.shape[1] != F or psd_mask.shape[0] not in (1, N):
                raise ValueError("psd_mask shape must be [F] or [N,F] with F=T//2+1")
            if psd_mask.shape[0] == 1:
                mask = psd_mask.to(device=device, dtype=dtype).repeat(N, 1)
            else:
                mask = psd_mask.to(device=device, dtype=dtype)
        else:
            raise ValueError("psd_mask must be rank-1 or rank-2 tensor")

        def _sample_time_from_mask_b(mask_batch: torch.Tensor) -> torch.Tensor:
            # mask_batch: [N, F]
            N_b, F_b = mask_batch.shape
            real = torch.randn(N_b, F_b, device=device, dtype=dtype)
            imag = torch.randn(N_b, F_b, device=device, dtype=dtype)
            imag[:, 0] = 0.0
            if T % 2 == 0:
                imag[:, -1] = 0.0
            X = (real + 1j * imag) * mask_batch
            y = torch.fft.irfft(X, n=T, dim=1)
            return y

        i_noise = _sample_time_from_mask_b(mask)
        q_noise = _sample_time_from_mask_b(mask)
        delta = torch.stack([i_noise, q_noise], dim=1)

        if spec_eps is not None:
            l2 = delta.pow(2).flatten(1).sum(dim=1).sqrt().clamp_min(1e-12)
            scale = (float(spec_eps) / l2).view(N, 1, 1)
            delta = delta * scale
        else:
            if jnr_db is None:
                raise ValueError("Provide either spec_eps or jnr_db to set noise amplitude.")
            rms_x = x.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()
            rms_delta = delta.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-12)
            jnr = 10.0 ** (jnr_db / 20.0)
            target_rms = rms_x / jnr
            delta = delta * (target_rms / rms_delta)

    else:
        raise NotImplementedError(f"Unknown spec_type: {spec_type}")

    x_adv = torch.clamp(x + delta, min=clip_min, max=clip_max)
    return x_adv
