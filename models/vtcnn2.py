"""
PyTorch implementation of VT-CNN2 (Virginia Tech CNN 2) for Automatic Modulation Classification.

Reference: https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb

Original Keras architecture:
    Reshape -> ZeroPad2D(0,2) -> Conv2D(256, 1x3) -> Dropout(0.5)
    -> ZeroPad2D(0,2) -> Conv2D(80, 2x3) -> Dropout(0.5)
    -> Flatten -> Dense(256) -> Dropout(0.5) -> Dense(num_classes) -> Softmax
"""

import torch
import torch.nn as nn


class VTCNN2(nn.Module):
    """
    VT-CNN2 model for modulation classification.

    Input: [batch, 2, 128] (I/Q signals)
    Output: (logits [batch, num_classes], regu_list [])
    """

    def __init__(self, num_classes=11, dropout_rate=0.5):
        super(VTCNN2, self).__init__()
        self.num_classes = num_classes

        # Conv1: input (1, 2, 128) -> pad to (1, 2, 132) -> Conv2d(256, kernel=(1,3)) -> (256, 2, 130)
        self.pad1 = nn.ZeroPad2d((2, 2, 0, 0))  # pad left/right by 2 on width (time) dim
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), bias=True)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        # Conv2: pad to (256, 2, 134) -> Conv2d(80, kernel=(2,3)) -> (80, 1, 132)
        self.pad2 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3), bias=True)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        # Compute flattened size: after conv2 output is (80, 1, 132)
        # pad1: (1,2,128) -> (1,2,132), conv1(1,3) valid on padded -> (256,2,130)
        # pad2: (256,2,130) -> (256,2,134), conv2(2,3) valid on padded -> (80,1,132)
        self._flat_size = 80 * 1 * 132

        self.fc1 = nn.Linear(self._flat_size, 256)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [batch, 2, 128]
        Returns:
            logits: [batch, num_classes]
            regu: [] (empty, for AWN training compatibility)
        """
        # [batch, 2, 128] -> [batch, 1, 2, 128]
        x = x.unsqueeze(1)

        x = self.pad1(x)
        x = self.drop1(self.relu1(self.conv1(x)))

        x = self.pad2(x)
        x = self.drop2(self.relu2(self.conv2(x)))

        x = x.reshape(x.size(0), -1)

        x = self.drop3(self.relu3(self.fc1(x)))
        logits = self.fc2(x)

        return logits, []


if __name__ == "__main__":
    model = VTCNN2(num_classes=11)
    x = torch.randn(32, 2, 128)
    out, regu = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}, Params: {sum(p.numel() for p in model.parameters()):,}")
