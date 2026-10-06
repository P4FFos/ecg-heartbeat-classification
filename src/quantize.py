from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.dataset import BeatDataset
from src.model import ECGNet
from src.train import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["N", "S", "V", "F"]


def quantize_weights(model, bits=8):
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            scale = param.abs().max() / qmax
            if scale == 0:
                continue
            param.copy_(torch.round(param / scale).clamp(-qmax, qmax) * scale)
    return model


def report(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    print(f"{name:<10} accuracy {acc:.4f}  macro F1 {macro:.4f}  "
          f"per-class {dict(zip(CLASSES, np.round(per_class, 3)))}")
    row = {"model": name, "accuracy": acc, "macro_f1": macro}
    row.update({f"f1_{cls}": score for cls, score in zip(CLASSES, per_class)})
    return acc, macro, row


def main():
    val_ds = BeatDataset("val")
    val_loader = DataLoader(val_ds, batch_size=512)

    state = torch.load(PROJECT_ROOT / "results" / "model.pt", map_location="cpu")

    fp32 = ECGNet(channels=(8, 16, 32))
    fp32.load_state_dict(state)
    pred, true = evaluate(fp32, val_loader)
    acc_fp32, macro_fp32, row_fp32 = report(true, pred, "FP32")

    int8 = ECGNet(channels=(8, 16, 32))
    int8.load_state_dict(state)
    quantize_weights(int8, bits=8)
    pred, true = evaluate(int8, val_loader)
    acc_int8, macro_int8, row_int8 = report(true, pred, "INT8")

    print(f"\ndifference: accuracy {acc_int8 - acc_fp32:+.4f}  "
          f"macro F1 {macro_int8 - macro_fp32:+.4f}")

    results = pd.DataFrame([row_fp32, row_int8])
    out_path = PROJECT_ROOT / "results" / "quantize_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nsaved results to {out_path}")


if __name__ == "__main__":
    main()