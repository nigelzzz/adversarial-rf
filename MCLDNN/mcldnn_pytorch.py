"""
PyTorch implementation of MCLDNN (Multi-Channel Learning Deep Neural Network)
for Automatic Modulation Recognition.

Original paper: "A Spatiotemporal Multi-Channel Learning Framework for
Automatic Modulation Recognition" (IEEE WCL 2020)

Converted from Keras implementation for compatibility with PyTorch-based pipeline.
"""

import torch
import torch.nn as nn


class MCLDNN_PyTorch(nn.Module):
    """
    MCLDNN model architecture in PyTorch.

    Original Keras inputs:
        - input1: [batch, 2, 128, 1] - I/Q combined (2 rows, 128 time samples)
        - input2: [batch, 128, 1] - I channel only
        - input3: [batch, 128, 1] - Q channel only

    This PyTorch version takes single input [batch, 2, 128] and splits internally.
    Output: [batch, num_classes] - Classification logits (with empty regularization list)
    """

    def __init__(self, num_classes=11, dropout_rate=0.5):
        super(MCLDNN_PyTorch, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping
        # Conv1: 2D conv on combined IQ, kernel (2,8)
        self.conv1 = nn.Conv2d(1, 50, kernel_size=(2, 8), padding=(1, 4))
        self.bn1 = nn.BatchNorm2d(50)

        # Conv2, Conv3: 1D conv on I and Q channels separately, kernel 8
        self.conv2 = nn.Conv1d(1, 50, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(50)
        self.conv3 = nn.Conv1d(1, 50, kernel_size=8, padding=4)
        self.bn3 = nn.BatchNorm1d(50)

        # Conv4: 2D conv on concatenated I/Q, kernel (1,8)
        self.conv4 = nn.Conv2d(50, 50, kernel_size=(1, 8), padding=(0, 4))
        self.bn4 = nn.BatchNorm2d(50)

        # Conv5: 2D conv on final concatenated features, kernel (2,5)
        self.conv5 = nn.Conv2d(100, 100, kernel_size=(2, 5), padding=0)
        self.bn5 = nn.BatchNorm2d(100)

        self.relu = nn.ReLU()

        # Part-B: Temporal Characteristics Extraction (LSTM)
        # After conv5: [batch, 100, 1, 124] -> reshape to [batch, 124, 100]
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False)

        # Part-C: DNN Classifier
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.selu = nn.SELU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, 2, 128] (I/Q signals)

        Returns:
            logits: Classification logits [batch, num_classes]
            regu: Empty list (for compatibility with AWN training)
        """
        batch_size = x.shape[0]

        # Prepare inputs
        # x: [batch, 2, 128]
        x_iq = x.unsqueeze(1)  # [batch, 1, 2, 128]
        x_i = x[:, 0:1, :]     # [batch, 1, 128]
        x_q = x[:, 1:2, :]     # [batch, 1, 128]

        # Branch 1: Conv1 on combined IQ
        # [batch, 1, 2, 128] -> conv (kernel 2x8) -> [batch, 50, H, W]
        x1 = self.relu(self.bn1(self.conv1(x_iq)))
        x1 = x1[:, :, :2, :128]  # Ensure shape [batch, 50, 2, 128]

        # Branch 2: Conv2 on I channel
        # [batch, 1, 128] -> conv (kernel 8) -> [batch, 50, 128]
        x2 = self.relu(self.bn2(self.conv2(x_i)))
        x2 = x2[:, :, :128]  # Ensure shape [batch, 50, 128]

        # Branch 3: Conv3 on Q channel
        x3 = self.relu(self.bn3(self.conv3(x_q)))
        x3 = x3[:, :, :128]

        # Reshape x2, x3 for concatenation: [batch, 50, 1, 128]
        x2 = x2.unsqueeze(2)
        x3 = x3.unsqueeze(2)

        # Concatenate I and Q branches on height dim: [batch, 50, 2, 128]
        x_iq_cat = torch.cat([x2, x3], dim=2)

        # Conv4 on concatenated I/Q
        x_iq_cat = self.relu(self.bn4(self.conv4(x_iq_cat)))
        x_iq_cat = x_iq_cat[:, :, :, :128]  # Ensure width is 128

        # Concatenate x1 and x_iq_cat on channel dim: [batch, 100, 2, 128]
        x_combined = torch.cat([x1, x_iq_cat], dim=1)

        # Conv5: [batch, 100, 2, 128] -> [batch, 100, 1, 124]
        x_combined = self.relu(self.bn5(self.conv5(x_combined)))

        # Reshape for LSTM: [batch, 124, 100]
        out = x_combined.squeeze(2)  # [batch, 100, 124]
        out = out.permute(0, 2, 1)   # [batch, 124, 100]

        # LSTM layers
        out, _ = self.lstm1(out)  # [batch, 124, 128]
        out, _ = self.lstm2(out)  # [batch, 124, 128]

        # Take last timestep
        out = out[:, -1, :]  # [batch, 128]

        # DNN classifier
        out = self.dropout(self.selu(self.fc1(out)))
        out = self.dropout(self.selu(self.fc2(out)))
        out = self.fc3(out)

        # Return tuple (logits, regu_sum) for compatibility with AWN training code
        return out, []


def test_model():
    """Quick test to verify model shapes."""
    model = MCLDNN_PyTorch(num_classes=11)
    x = torch.randn(32, 2, 128)
    out, regu = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Regularization terms: {regu}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    test_model()
