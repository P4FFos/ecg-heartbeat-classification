from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.dataset import BeatDataset
from src.model import ECGNet, count_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 8
LR = 1e-3
SEED = 42


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        p = torch.exp(-ce)
        return ((1 - p) ** self.gamma * ce).mean()


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x).argmax(1).numpy())
            targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = BeatDataset("train")
    val_ds = BeatDataset("val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    model = ECGNet(channels=(8, 16, 32))
    print(f"{count_parameters(model)} parameters")

    counts = train_ds.class_counts()
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    out_path = PROJECT_ROOT / "results" / "model.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch = -1.0, -1

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        pred, true = evaluate(model, val_loader)
        score = f1_score(true, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        per_class = f1_score(true, pred, average=None, zero_division=0)

        print(
            f"epoch {epoch:>2}  loss {total_loss / len(train_ds):.4f}  "
            f"NSV {score:.4f}  per-class {np.round(per_class, 3)}",
            flush=True,
        )

        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), out_path)
        elif epoch - best_epoch >= PATIENCE:
            print(f"early stop at epoch {epoch}")
            break

    print(f"best NSV macro F1 {best:.4f} at epoch {best_epoch}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
