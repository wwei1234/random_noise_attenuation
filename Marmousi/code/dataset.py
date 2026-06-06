import os

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, samples_dir, labels_dir):
        self.samples_dir = samples_dir
        self.labels_dir = labels_dir
        self.samples = sorted(os.listdir(samples_dir))
        self.labels = sorted(os.listdir(labels_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.samples_dir, self.samples[idx])
        label_path = os.path.join(self.labels_dir, self.labels[idx])

        sample = np.load(sample_path)
        label = np.load(label_path)

        sample = np.expand_dims(sample, axis=0)
        label = np.expand_dims(label, axis=0)

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return sample, label
