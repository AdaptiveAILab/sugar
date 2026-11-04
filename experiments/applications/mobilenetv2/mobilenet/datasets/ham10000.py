import csv
import numpy as np
from PIL import Image
from pathlib import Path
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader

from mobilenet.utils import build_transforms, build_loaders

# ----------------------
# Dataset HAM10000
# ----------------------

# Place these under args.data_root:
#   HAM10000_metadata.csv
#   HAM10000_images_part_1/  *.jpg
#   HAM10000_images_part_2/  *.jpg

HAM_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM_LABEL_TO_ID = {c: i for i, c in enumerate(HAM_LABELS)}
NUM_CLASSES = len(HAM_LABELS)

class HAM10000Dataset(Dataset):
    def __init__(self, samples, indices, transform):
        # samples: list[ (Path_to_img, int_label) ]
        self.samples = samples
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img_path, target = self.samples[idx]
        with Image.open(img_path).convert("RGB") as im:
            img = im.copy()
        if self.transform:
            img = self.transform(img)
        return img, target


def _require_ham_files(root: Path) -> tuple[Path, Path, Path]:
    meta = root / "HAM10000_metadata.csv"
    p1 = root / "HAM10000_images_part_1"
    p2 = root / "HAM10000_images_part_2"
    missing = [p for p in [meta, p1, p2] if not p.exists()]
    if missing:
        msg = (
            "HAM10000 files not found under "
            f"{root}. Expected: HAM10000_metadata.csv, HAM10000_images_part_1/, HAM10000_images_part_2/.\n"
            "Download from the official source (Harvard Dataverse DOI 10.7910/DVN/DBW86T) "
            "and extract here."
        )
        raise FileNotFoundError(msg)
    return meta, p1, p2

def _read_ham_metadata(meta_csv: Path) -> list:
    records = []  # (image_id, dx_label_id)
    with open(meta_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row["image_id"]
            dx = row["dx"].strip()
            if dx not in HAM_LABEL_TO_ID:
                continue
            records.append((img_id, HAM_LABEL_TO_ID[dx]))
    return records


def prepare_ham10000(data_root: Path, train_frac: float, val_frac: float,
                     seed: int) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    data_root.mkdir(parents=True, exist_ok=True)
    meta_csv, p1, p2 = _require_ham_files(data_root)

    recs = _read_ham_metadata(meta_csv)
    samples = []
    for img_id, y in recs:
        p = p1 / f"{img_id}.jpg"
        if not p.exists():
            p = p2 / f"{img_id}.jpg"
        if not p.exists():
            continue
        samples.append((p, y))

    if not samples:
        raise RuntimeError("No HAM10000 images found after scanning metadata and part folders.")

    # Stratified split: 70/10/20 by class (defaults train_frac=0.7, val_frac=0.1)
    rng = np.random.default_rng(seed)
    by_class = {}
    for i, (_, y) in enumerate(samples):
        by_class.setdefault(y, []).append(i)
    train_idx, val_idx, test_idx = [], [], []
    for y, idxs in by_class.items():
        idxs = np.array(idxs, dtype=np.int64)
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        n_test = n - n_train - n_val
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train+n_val])
        test_idx.extend(idxs[-n_test:])

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)
    return samples, train_idx, val_idx, test_idx

def get_data(args: Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    samples, trn, val, tst = prepare_ham10000(args.data_root, args.train_frac, args.val_frac, args.seed)
    train_tfms, eval_tfms = build_transforms(args.img_size)
    train_ds = HAM10000Dataset(samples, trn, train_tfms)
    val_ds = HAM10000Dataset(samples, val, eval_tfms)
    test_ds = HAM10000Dataset(samples, tst, eval_tfms)
    return build_loaders(train_ds, val_ds, test_ds, args.batch_size, args.num_workers)
