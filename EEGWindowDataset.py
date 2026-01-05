import torch
from torch.utils.data import Dataset

class EEGWindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows.float()
        self.labels = labels.long()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]