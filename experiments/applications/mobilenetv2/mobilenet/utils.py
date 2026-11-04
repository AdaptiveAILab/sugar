import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# ----------------------
# Data
# ----------------------
# Apply ImageNet1k statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, eval_tfms


def build_loaders(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset,
                  batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ----------------------
# Utilities
# ----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def human_time(s: float) -> str:
    m = int(s // 60)
    s = int(s % 60)
    return f"{m}m{s:02d}s"

# ----------------------
# Training
# ----------------------
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        # output: [B, C], target: [B]
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        batch_size = target.size(0)
        return 100.0 * correct / batch_size

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)

            batch_size = targets.size(0)
            # accumulate weighted by batch size to get dataset-level mean
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, scaler=None) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        batch_size = targets.size(0)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        acc = accuracy(outputs, targets)

        total_loss += loss.item() * batch_size
        total_acc += acc * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc


# ----------------------
# Transfer learning
# ----------------------
def freeze_all_but_classifier_(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

def unfreeze_last_n_params_(model: nn.Module, unfreeze_last: int) -> None:
    params = [p for p in model.parameters()]
    for p in params[-unfreeze_last:]:
        p.requires_grad = True