import numpy as np
from PIL import Image
from pathlib import Path
from scipy.io import loadmat
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader

from mobilenet.utils import build_transforms, build_loaders

# ----------------------
# Dataset Oxford102
# ----------------------

# Place these under args.data_root:
#   jpg/             *.jpg (extracted from 102flowers.tar)
#   imagelabels.mat
#   setid.mat

NUM_CLASSES = 102

DEFAULTS = {
    "weight_decay": 1e-5,
}


class Flowers102Dataset(Dataset):
    def __init__(self, images_dir: Path, labels_1based: np.ndarray, indices_1based: np.ndarray, transform):
        self.images_dir = images_dir
        self.labels = labels_1based.astype(np.int64) - 1  # -> 0..101
        self.indices = [int(i) for i in indices_1based]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img_path = self.images_dir / f"image_{idx:05d}.jpg"
        with Image.open(img_path).convert("RGB") as im:
            img = im.copy()
        target = int(self.labels[idx - 1])  # idx is 1‑based
        if self.transform:
            img = self.transform(img)
        return img, target


def _require_oxford_files(root: Path) -> tuple[Path, Path, Path]:
    images_dir = root / "jpg"
    labels_mat = root / "imagelabels.mat"
    setid_mat = root / "setid.mat"
    missing = [p for p in [images_dir, labels_mat, setid_mat] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Oxford102 files not found under {root}. "
            "Expected: jpg/ (extracted images), imagelabels.mat, setid.mat. "
            "See README for download instructions."
        )
    return images_dir, labels_mat, setid_mat


def prepare_oxford102(data_root: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images_dir, labels_mat, setid_mat = _require_oxford_files(data_root)

    labels = loadmat(labels_mat.as_posix())["labels"].squeeze()
    splits = loadmat(setid_mat.as_posix())
    train = splits["trnid"].squeeze()
    val   = splits["valid"].squeeze()
    test  = splits["tstid"].squeeze()
    return images_dir, labels, train, val, test


def get_data(args: Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    images_dir, labels, train, val, test = prepare_oxford102(args.data_root)
    train_tfms, eval_tfms = build_transforms(args.img_size)
    train_ds = Flowers102Dataset(images_dir, labels, train, train_tfms)
    val_ds = Flowers102Dataset(images_dir, labels, val, eval_tfms)
    test_ds = Flowers102Dataset(images_dir, labels, test, eval_tfms)
    return build_loaders(train_ds, val_ds, test_ds, args.batch_size, args.num_workers)
