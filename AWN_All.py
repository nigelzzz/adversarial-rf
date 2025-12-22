import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np

# ======== path setup start ========
Device='cpu'
TopN=50
CLS_WEIGHT_PATH='/content/drive/MyDrive/AWN_Classifier/AWN_CLS_best_acc.pth'
DETECTOR_WEIGHT_PATH='/content/drive/MyDrive/AWN_Detector/Detector_CNN_best.pth'
SAMPLES_PATH='/content/drive/MyDrive/AWN-main/data_atk/clean_220_samples.pt'
DIV_THRESHOLD=0.004468164592981338

# ======== functions declaration start ========
# FFT/IFFT function
def filter_top_components_torch(data, top_n):
    samples, components, length = data.shape
    filtered_data = torch.zeros_like(data, dtype=torch.complex64)

    for i in range(samples):
        for j in range(components):
            fft_result = torch.fft.fft(data[i, j])

            _, indices = torch.topk(torch.abs(fft_result), top_n)

            filtered_fft = torch.zeros_like(fft_result)
            filtered_fft[indices] = fft_result[indices]

            filtered_data[i, j] = torch.fft.ifft(filtered_fft)

    return filtered_data.real

# Normalization function
def normalize_data(data):
    return (data + 0.02) / 0.04

# KL divergence function
def kl_divergence(p, q):
    kl_divs = torch.zeros(p.size(0))
    for i in range(p.size(0)):
        p_sample = F.softmax(p[i], dim=-1)
        q_sample = F.softmax(q[i], dim=-1)
        kl_divs[i] = (p_sample * (p_sample.log() - q_sample.log())).sum()
    return kl_divs

# ======== model declaration start ========
class RFSignalAutoEncoder(nn.Module):
    def __init__(self):
        super(RFSignalAutoEncoder, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1)
        self.enc_bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.3)  # Lower dropout rate
        self.enc_conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.enc_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=2, dilation=2)
        self.enc_bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        self.enc_conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=4, dilation=4)
        self.enc_bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)

        # Attention Layer
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.Sigmoid()
        )

        # Decoder
        self.dec_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn1 = nn.BatchNorm1d(64)
        self.dec_conv2 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn2 = nn.BatchNorm1d(32)
        self.dec_conv3 = nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_bn3 = nn.BatchNorm1d(16)
        self.dec_conv4 = nn.ConvTranspose1d(16, 2, kernel_size=3, stride=1, padding=1)

        self.skip1 = nn.Conv1d(2, 128, kernel_size=1, stride=8)

    def forward(self, x):
        # Encoding
        x1 = self.dropout1(F.relu(self.enc_bn1(self.enc_conv1(x))))
        x2 = self.dropout2(F.relu(self.enc_bn2(self.enc_conv2(x1))))
        x3 = self.dropout3(F.relu(self.enc_bn3(self.enc_conv3(x2))))
        x4 = self.dropout4(F.relu(self.enc_bn4(self.enc_conv4(x3))))

        # Apply attention
        attention_weights = self.attention(x4)
        x4 = x4 * attention_weights

        # Decoding
        x4 += self.skip1(x)
        x5 = F.relu(self.dec_bn1(self.dec_conv1(x4)))
        x6 = F.relu(self.dec_bn2(self.dec_conv2(x5)))
        x7 = F.relu(self.dec_bn3(self.dec_conv3(x6)))
        x8 = self.dec_conv4(x7)

        return x8

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

        self.conv_even = lambda x: x[:, :, ::2]
        self.conv_odd = lambda x: x[:, :, 1::2]

    def forward(self, x):
        """
        returns the odd and even part
        :param x:
        :return: x_even, x_odd
        """
        return self.conv_even(x), self.conv_odd(x)


class Operator(nn.Module):
    def __init__(self, in_planes, kernel_size=3, dropout=0.):
        super(Operator, self).__init__()

        pad = (kernel_size - 1) // 2 + 1

        self.operator = nn.Sequential(
            nn.ReflectionPad1d(pad),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(in_planes, in_planes,
                      kernel_size=(kernel_size,), stride=(1,)),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Operator as Predictor() or Updator()
        :param x:
        :return: P(x) or U(x)
        """
        x = self.operator(x)
        return x


class LiftingScheme(nn.Module):
    def __init__(self, in_planes, kernel_size=3):
        super(LiftingScheme, self).__init__()

        self.split = Splitting()

        self.P = Operator(in_planes, kernel_size)
        self.U = Operator(in_planes, kernel_size)

    def forward(self, x):
        """
        Implement Lifting Scheme
        :param x:
        :return: c: approximation coefficient
                 d: details coefficient
        """
        (x_even, x_odd) = self.split(x)
        c = x_even + self.U(x_odd)
        d = x_odd - self.P(c)
        return c, d

class LevelTWaveNet(nn.Module):
    def __init__(self, in_planes, kernel_size, regu_details, regu_approx):
        super(LevelTWaveNet, self).__init__()
        self.regu_details = regu_details
        self.regu_approx = regu_approx
        self.wavelet = LiftingScheme(in_planes, kernel_size=kernel_size)

    def forward(self, x):
        """
        Conduct decomposition and calculate regularization terms
        :param x:
        :return: approx component, details component, regularization terms
        """
        global regu_d, regu_c
        (L, H) = self.wavelet(x)  # 10 9 128
        approx = L
        details = H
        if self.regu_approx + self.regu_details != 0.0:
            if self.regu_details:
                regu_d = self.regu_details * H.abs().mean()
            # Constrain on the approximation
            if self.regu_approx:
                regu_c = self.regu_approx * torch.dist(approx.mean(), x.mean(), p=2)
            if self.regu_approx == 0.0:
                # Only the details
                regu = regu_d
            elif self.regu_details == 0.0:
                # Only the approximation
                regu = regu_c
            else:
                # Both
                regu = regu_d + regu_c

            return approx, details, regu


class AWN(nn.Module):
    def __init__(self,
                 num_classes=11,
                 num_levels=1,
                 in_channels=64,
                 kernel_size=3,
                 latent_dim=320,
                 regu_details=0.01,
                 regu_approx=0.01):
        super(AWN, self).__init__()

        self.num_classes = num_classes
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_channels = self.in_channels * (self.num_levels + 1)
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.regu_details = regu_details
        self.regu_approx = regu_approx

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((3, 3, 0, 0)),
            # Call a 2d Conv to integrate I, Q channels
            nn.Conv2d(1, self.in_channels,
                      kernel_size=(2, 7), stride=(1,), bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.in_channels,
                      kernel_size=(5,), stride=(1,), padding=(2,), bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.levels = nn.ModuleList()

        for i in range(self.num_levels):
            self.levels.add_module(
                'level_' + str(i),
                LevelTWaveNet(self.in_channels,
                              self.kernel_size,
                              self.regu_details,
                              self.regu_approx)
            )

        self.SE_attention_score = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels // 4, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels // 4, self.out_channels, bias=False),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(self.out_channels, self.latent_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(self.latent_dim, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # x:[N, 2, T] -> [N, 1, 2, T]
        x = self.conv1(x)
        x = x.squeeze(2)  # x:[N, C, 1, T] -> [N, C, T]
        x = self.conv2(x)
        regu_sum = []  # List of constrains on details and mean
        det = []  # List of averaged pooled details

        for l in self.levels:
            x, details, regu = l(x)
            regu_sum += [regu]
            det += [self.avgpool(details)]
        aprox = self.avgpool(x)
        det += [aprox]

        x = torch.cat(det, 1)
        x = x.view(-1, x.size()[1])
        x = torch.mul(self.SE_attention_score(x), x)

        logit = self.fc(x)

        return logit

# ======== perform Detector + FFT/IFFT + Classifier start ========
# Load CLS Model & Weight
device = Device
model = AWN().to(device)
model_path = CLS_WEIGHT_PATH
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Load Detector Model & Weight
detector_path = DETECTOR_WEIGHT_PATH
detector = RFSignalAutoEncoder().to(device)
detector.load_state_dict(torch.load(detector_path, map_location=torch.device('cpu')))
detector.eval()

# Load Inputs
samples_path = SAMPLES_PATH
samples = torch.load(samples_path)
inputs, labels = zip(*samples)
inputs_tensor = torch.cat(inputs)
labels_tensor = torch.cat(labels)

# Normalize inputs before passing through the detector
normalized_inputs = normalize_data(inputs_tensor.to(device))

# Input go through Detector and calucalte the Divergence
with torch.no_grad():
    reconstructed = detector(normalized_inputs)
    kl_divs = kl_divergence(normalized_inputs, reconstructed).cpu()

threshold = DIV_THRESHOLD

pass_indices = torch.where(kl_divs <= threshold)[0]
drop_indices = torch.where(kl_divs > threshold)[0]  # Calculate dropped samples

# If div value is less than 90th div of clean data
if len(pass_indices) > 0:
    # Prepare DataLoader for samples passed by the detector
    passed_inputs = torch.stack([inputs_tensor[j] for j in pass_indices])
    passed_labels = torch.stack([labels_tensor[j] for j in pass_indices])
    dataset = TensorDataset(passed_inputs, passed_labels)
    loader = DataLoader(dataset, batch_size=50, shuffle=False)

    # Model evaluation
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = normalize_data(inputs)  # Ensure normalization here as well

        filtered_sample = filter_top_components_torch(inputs, TopN)

        outputs = model(filtered_sample)
        _, predicted = torch.max(outputs, 1)

        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions * 100
    print(f'Accuracy : {accuracy:.2f}%')
else:
    print(f'All samples were dropped due to high KL divergence')

print(f'Dropped {drop_indices.size(0)} samples due to high KL divergence')
