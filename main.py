import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import PKLDataset
from model import MyCNN


# -----------------------------
# Config
# -----------------------------

RESUME_TRAINING = True          # <-- set to False if you ever want to start fresh
CHECKPOINT_PATH = "checkpoint.pth"
BEST_MODEL_PATH = "model.pth"


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Data Loaders for PKL data
# -----------------------------

def get_dataloaders(train_pkl: str = "train.pkl",
                    val_pkl: str = "val.pkl",
                    batch_size: int = 64):

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # Validation: just resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_dataset = PKLDataset(train_pkl, transform=train_transform)
    val_dataset   = PKLDataset(val_pkl,   transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    print(f"Loaded {len(train_dataset)} training images, "
          f"{len(val_dataset)} validation images.")

    return train_loader, val_loader


# -----------------------------
# Train & Evaluate
# -----------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


# -----------------------------
# Main Training Loop (with resume)
# -----------------------------

def main():
    set_seed(42)
    
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass  # safe to ignore if unsupported
        
# Prefer MPS (Apple GPU) → then CUDA → fallback to CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using device: MPS (Apple GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
        print("Using device:", device)

    train_loader, val_loader = get_dataloaders(
        train_pkl="train.pkl",
        val_pkl="val.pkl",
        batch_size=64
    )

    num_classes = 15  # per assignment: 15 classes
    model = MyCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,    # reduce LR by 0.5
        patience=2,    # wait 2 epochs with no val_loss improvement
    )

    num_epochs = 60
    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = 7

    # -------------------------
    # Resume logic
    # -------------------------
    start_epoch = 1
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"🔁 Found checkpoint at {CHECKPOINT_PATH}. Resuming training...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        last_epoch = checkpoint.get("epoch", 0)
        start_epoch = last_epoch + 1

        print(f"Resumed from epoch {last_epoch}, best_val_acc={best_val_acc:.4f}, "
              f"epochs_no_improve={epochs_no_improve}")
    else:
        print("No checkpoint found or resume disabled. Starting from scratch.")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr != old_lr:
            print(f"⚠️  Learning rate reduced from {old_lr:.6f} → {new_lr:.6f}")

        # Check for improvement (for best model)
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✅ New best model saved with Val Acc: {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")

        # ---------------------
        # Save checkpoint every epoch
        # ---------------------
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "epochs_no_improve": epochs_no_improve,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"  💾 Checkpoint saved at epoch {epoch}.")

        # Early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Best model weights saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
