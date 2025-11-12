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

    We REMAP labels to 0..num_classes-1 for PyTorch.
    """
    def __init__(self, pkl_path, transform=None):
        super().__init__()

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.images = data["images"]      # (N, 64, 64, 3)
        original_labels = data["labels"]  # list, length N
        self.transform = transform

        assert len(self.images) == len(original_labels), \
            "Images and labels length mismatch"

        # ---- NEW PART: remap labels to 0..(K-1) ----
        unique_labels = sorted(set(original_labels))
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        # Optional: print to verify once
        print("Unique labels in this file:", unique_labels)
        print("Number of unique classes:", len(unique_labels))

        self.labels = [self.label_map[l] for l in original_labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]     # numpy array HWC (64,64,3)
        label = self.labels[idx]   # already remapped to 0..K-1

        # Ensure uint8 then convert to PIL Image
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        img = Image.fromarray(img)  # PIL image

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label
