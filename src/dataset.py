from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(PROJECT_ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


class BeatDataset(Dataset):
    def __init__(self, split, augment=None, cfg=None):
        if split != "train" and augment is not None:
            raise ValueError("augmentation is training-only (report 3.7)")

        cfg = cfg or load_config()
        self.classes = cfg["labels"]["classes"]
        label_to_index = {c: i for i, c in enumerate(self.classes)}

        processed = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
        data = np.load(processed / f"{split}.npz", allow_pickle=True)

        self.X = data["X"].astype(np.float32)
        self.y = np.array([label_to_index[l] for l in data["y"]], dtype=np.int64)
        self.record = data["record"]
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment is not None:
            x = self.augment(x)
        return torch.from_numpy(x).unsqueeze(0), int(self.y[i])

    def class_counts(self):
        return np.bincount(self.y, minlength=len(self.classes))
