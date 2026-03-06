"""
1D Residual Network for Automatic Modulation Classification.

Input: [batch, 2, T] (I/Q signals, T=128 or T=1024)
Output: (logits [batch, num_classes], regu_list [])
"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResNet1D(nn.Module):
    """
    1D ResNet for modulation classification.

    Architecture: stem Conv1d(2->64) -> 4 ResidualBlock1D -> AdaptiveAvgPool1d(1)
                  -> FC(64->128->num_classes)
    """

    def __init__(self, num_classes=11):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock1D(64)
        self.layer2 = ResidualBlock1D(64)
        self.layer3 = ResidualBlock1D(64)
        self.layer4 = ResidualBlock1D(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
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
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits, []


if __name__ == "__main__":
    model = ResNet1D(num_classes=11)
    for T in [128, 1024]:
        x = torch.randn(32, 2, T)
        out, regu = model(x)
        print(f"Input: {x.shape}, Output: {out.shape}, "
              f"Params: {sum(p.numel() for p in model.parameters()):,}")
