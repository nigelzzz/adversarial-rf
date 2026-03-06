"""
CNN-LSTM for Automatic Modulation Classification.

Uses a 1D CNN front-end for feature extraction, then BiLSTM for temporal modeling.
This hybrid architecture handles the small amplitude IQ signals effectively.

Input: [batch, 2, T] (I/Q signals, T=128 or T=1024)
Output: (logits [batch, num_classes], regu_list [])
"""

import torch
import torch.nn as nn


class LSTM_AMC(nn.Module):
    """
    CNN-BiLSTM model for modulation classification.

    Architecture: Conv1d feature extractor -> BiLSTM -> FC head.
    The CNN front-end maps raw IQ (2 channels) to a richer feature space
    before the LSTM, solving the vanishing-signal problem with tiny IQ amplitudes.
    """

    def __init__(self, num_classes=11, hidden_size=128, num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        feat_dim = 64

        # CNN feature extractor: [batch, 2, T] -> [batch, feat_dim, T]
        self.cnn = nn.Sequential(
            nn.Conv1d(2, feat_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # BiLSTM produces 2*hidden_size per direction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 2, T]
        Returns:
            logits: [batch, num_classes]
            regu: [] (empty, for AWN training compatibility)
        """
        # CNN feature extraction: [batch, 2, T] -> [batch, feat_dim, T]
        x = self.cnn(x)
        # [batch, feat_dim, T] -> [batch, T, feat_dim]
        x = x.permute(0, 2, 1)
        # LSTM: [batch, T, 2*hidden_size]
        out, _ = self.lstm(x)
        # Use last time step output
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits, []


if __name__ == "__main__":
    model = LSTM_AMC(num_classes=11)
    for T in [128, 1024]:
        x = torch.randn(32, 2, T)
        out, regu = model(x)
        print(f"Input: {x.shape}, Output: {out.shape}, "
              f"Params: {sum(p.numel() for p in model.parameters()):,}")
