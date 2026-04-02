from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 32
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 2
DEFAULT_INDEX_COLUMN = "train_id_original"


class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds = hf_split
        self.transform = transform
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.ds[real_idx]
        img = ex["image"]
        y = int(ex["label"])
        x = self.transform(img) if self.transform else img
        return x, y, real_idx


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_num_workers(requested: int) -> int:
    # Keep notebook kernels stable by disabling worker processes there.
    if "ipykernel" in sys.modules and int(requested) > 0:
        return 0
    return int(requested)


def build_transform(image_size: int = DEFAULT_IMAGE_SIZE):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])


def find_project_root(start_path: Path | None = None) -> Path:
    """Find the project root containing scripts/ and ArtBench-10/."""
    start = Path(start_path) if start_path is not None else Path(__file__).resolve().parent
    candidates = [start, *start.parents]

    for candidate in candidates:
        scripts_ok = (candidate / "scripts" / "artbench_local_dataset.py").exists()
        data_ok = (candidate / "ArtBench-10").exists()
        if scripts_ok and data_ok:
            return candidate

    raise FileNotFoundError(
        "Could not find project root with scripts/artbench_local_dataset.py and ArtBench-10/."
    )


def load_artbench_train_split(project_root: Path):
    scripts_dir = project_root / "scripts"
    kaggle_root = project_root / "ArtBench-10"

    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from artbench_local_dataset import load_kaggle_artbench10_splits

    hf_ds = load_kaggle_artbench10_splits(kaggle_root)
    train_hf = hf_ds["train"]
    class_names = list(train_hf.features["label"].names)
    return train_hf, class_names


def load_ids_from_training_csv(csv_path: Path, index_column: str = DEFAULT_INDEX_COLUMN) -> list[int]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"training csv not found: {csv_path}"
        )

    ids: list[int] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if index_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column {index_column!r} not present in {csv_path}. "
                f"Available: {reader.fieldnames}"
            )
        for row in reader:
            value = str(row.get(index_column, "")).strip()
            if value:
                ids.append(int(value))

    if not ids:
        raise ValueError(f"No ids found in {csv_path} column {index_column!r}")

    return ids


def build_csv_subset_train_loader(
    train_hf,
    csv_path: Path,
    index_column: str = DEFAULT_INDEX_COLUMN,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    shuffle: bool = True,
):
    transform = build_transform(image_size=image_size)
    train_ids = load_ids_from_training_csv(csv_path, index_column=index_column)

    train_ds = HFDatasetTorch(train_hf, transform=transform, indices=train_ids)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=safe_num_workers(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader


def setup_artbench_from_csv_subset(
    project_root: Path | None = None,
    training_csv_path: Path | None = None,
    index_column: str = DEFAULT_INDEX_COLUMN,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    shuffle: bool = True,
    seed: int = DEFAULT_SEED,
):
    """Resolve paths, load ArtBench train split, and return subset loader + metadata."""
    set_seed(seed)

    root = find_project_root(project_root)
    csv_path = Path(training_csv_path) if training_csv_path is not None else root / "src" / "training_20_percent.csv"

    train_hf, class_names = load_artbench_train_split(root)
    train_loader = build_csv_subset_train_loader(
        train_hf=train_hf,
        csv_path=csv_path,
        index_column=index_column,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return {
        "project_root": root,
        "training_csv_path": csv_path,
        "train_hf": train_hf,
        "class_names": class_names,
        "train_loader": train_loader,
    }
