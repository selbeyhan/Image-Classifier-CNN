import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class PKLDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        super().__init__()

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # data should contain "images" and "labels"
        self.images = data["images"]  # shape: (N, 64, 64, 3)
        self.labels = data["labels"]  # shape: (N,)
        self.transform = transform

        # convert to uint8 if necessary
        if self.images.dtype != np.uint8:
            self.images = self.images.astype(np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Convert numpy HWC -> PyTorch CHW
        img = torch.tensor(img).permute(2, 0, 1)  # (3, 64, 64)

        img = img.float() / 255.0  # normalize to 0–1

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label)
