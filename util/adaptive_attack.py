"""
Adaptive attack wrapper: chains FFT Top-K defense into the model so that
adversarial attacks can optimize *through* the defense.

Key insight: fft_topk_denoise in util/defense.py is fully differentiable
(torch.fft.fft -> topk -> scatter_ -> ifft). The topk operation itself has
useful gradients through the selected components.
"""

from typing import Optional

import torch
import torch.nn as nn

from util.defense import fft_topk_denoise


class DefendedModel01Wrapper(nn.Module):
    """
    Wrapper that chains FFT Top-K defense before classification.

    Accepts [0,1] inputs (like Model01Wrapper) and:
    1. Converts to IQ space [-1,1] (or via minmax mapping)
    2. Applies fft_topk_denoise
    3. Forwards through base classifier
    4. Returns logits

    Compatible with torchattacks (same interface as Model01Wrapper).
    """

    def __init__(self, base_model: nn.Module, topk: int = 50):
        super().__init__()
        self.base = base_model
        self.topk = topk
        self._use_minmax: bool = False
        self._a: Optional[torch.Tensor] = None
        self._b: Optional[torch.Tensor] = None

    def set_minmax(self, a: torch.Tensor, b: torch.Tensor):
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
            raise ValueError(f"Unexpected input shape: {tuple(x01.shape)}")

        # Convert to IQ space
        if self._use_minmax:
            x_iq = x01 * self._b + self._a
        else:
            x_iq = 2.0 * x01 - 1.0

        # Apply differentiable FFT Top-K defense
        x_defended = fft_topk_denoise(x_iq, self.topk)

        # Classify
        logits, _ = self.base(x_defended)
        return logits
