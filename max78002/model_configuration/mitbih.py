# MIT-BIH beats from data/processed for ai8x-training
# Class weights are the same as in src/train.py

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["N", "S", "V", "F"]
CLIP_SIGMA = 12.0
CLASS_WEIGHTS = [0.2778, 13.0663, 3.5245, 25.1355]


class MITBIHBeats(Dataset):
    def __init__(self, data_dir, splits, act_mode_8bit):
        label_to_index = {c: i for i, c in enumerate(CLASSES)}

        X, y = [], []
        self.sizes = {}
        for split in splits:
            data = np.load(Path(data_dir) / f"{split}.npz")
            X.append(data["X"])
            y += [label_to_index[label] for label in data["y"]]
            self.sizes[split] = len(data["y"])

        x = np.clip(np.round(np.concatenate(X) / CLIP_SIGMA * 128), -128, 127)
        if not act_mode_8bit:
            x = x / 128

        self.X = torch.from_numpy(x.astype(np.float32)).unsqueeze(1)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def mitbih_get_datasets(data, load_train=True, load_test=True):
    data_dir, args = data

    train_ds, test_ds = None, None
    if load_train:
        train_ds = MITBIHBeats(data_dir, ["train", "val"], args.act_mode_8bit)
        train_ds.valid_indices = list(range(train_ds.sizes["train"], len(train_ds)))
    if load_test:
        test_ds = MITBIHBeats(data_dir, ["test"], args.act_mode_8bit)

    return train_ds, test_ds


datasets = [
    {
        "name": "MITBIH",
        "input": (1, 200),
        "output": CLASSES,
        "weight": CLASS_WEIGHTS,
        "loader": mitbih_get_datasets,
    },
]
