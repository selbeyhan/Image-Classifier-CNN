import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MyCNN(nn.Module):
    """
    CNN for 64x64 RGB images, 15 classes.
    Uses ONLY Conv2d, Pooling, BatchNorm, Dropout, Flatten, Dense.
    (Matches assignment constraints.)
    """
    def __init__(self, num_classes: int = 15):
        super(MyCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # After 4 pools: 64 -> 32 -> 16 -> 8 -> 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(weight_path: str = "model.pth",
               device: str | torch.device = "cpu") -> MyCNN:
    """
    Load the trained model from model.pth.
    The instructor will call this.
    """
    if isinstance(device, str):
        device = torch.device(device)

    model = MyCNN(num_classes=15)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module,
            dataloader: DataLoader,
            device: str | torch.device = "cpu"):
    """
    Given a dataloader, return a list of predicted class indices.
    Instructor will call this on their hidden test set.
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    model.eval()

    all_preds = []

    for batch in dataloader:
        # Handle (images, labels) or just images
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())

    return all_preds
