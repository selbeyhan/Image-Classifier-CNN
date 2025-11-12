import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class PKLDataset(Dataset):
    """
    Dataset for Tiny ImageNet subset stored as a .pkl dict:
    {
        'images': np.ndarray of shape (N, 64, 64, 3),
        'labels': list of length N,
        'class_names': dict,
        'all_classes': list
    }
    """
    def __init__(self, pkl_path, transform=None):
        super().__init__()

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.images = data["images"]      # (N, 64, 64, 3)
        self.labels = data["labels"]      # list, length N
        self.transform = transform

        assert len(self.images) == len(self.labels), \
            "Images and labels length mismatch"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]     # numpy array HWC (64,64,3)
        label = self.labels[idx]   # usually an int

        # Ensure uint8 then convert to PIL Image
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        img = Image.fromarray(img)  # PIL image

        if self.transform is not None:
            img = self.transform(img)

        # CrossEntropyLoss expects targets as LongTensor
        label = torch.tensor(label, dtype=torch.long)

        return img, label
