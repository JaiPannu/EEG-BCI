import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class EEGWindowDataset(Dataset):
    def __init__(self, windows, labels):
        """
        windows: tensor [N, C, T]
        labels:  tensor [N]
        """
        self.windows = windows.float()
        self.labels = labels.long()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

class AlphaWaveDetector(nn.Module):
    def __init__(self, in_channels, sampling_rate):
        super().__init__()

        # Calculate convolution kernel length ≈ 1 cycle at 10 Hz
        alpha_center = 10.0
        cycle_length = int(sampling_rate / alpha_center)  # samples per cycle
        
        if cycle_length < 3:
            cycle_length = 3  # minimum safety

        # A bank of filters aimed at frequencies around alpha
        self.conv_alpha = nn.Conv1d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=cycle_length,
            padding="same",
            bias=False
        )

        # A tiny classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # global temporal pooling
            nn.Flatten(),
            nn.Linear(8, 1)
        )

        # Initialize filters to approximate sinusoidal shapes
        self._init_alpha_filters(alpha_center, sampling_rate)

    # Initialize conv filters as sine waves around alpha frequency
    def _init_alpha_filters(self, freq, fs):
        with torch.no_grad():
            for i in range(self.conv_alpha.weight.shape[0]):
                t = torch.arange(self.conv_alpha.weight.shape[-1]) / fs
                sine = torch.sin(2 * torch.pi * freq * t)
                # broadcast to in_channels
                self.conv_alpha.weight[i] = sine

    def forward(self, x):
        """
        x: EEG batch, shape [B, C, T]
        returns: logits for alpha presence [B, 1]
        """

        # Step 1: apply learnable alpha band filters
        filtered = self.conv_alpha(x)
        filtered = F.relu(filtered)

        # Step 2: compress time dimension → feature vector
        logits = self.classifier(filtered)

        return logits
