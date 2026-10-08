from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import accuracy_score, f1_score

from src.dataset import BeatDataset
from src.model import ECGNet

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NSTDB = PROJECT_ROOT / "data" / "mit-bih-noise-stress-test-database-1.0.0"
CLASSES = ["N", "S", "V", "F"]


def add_noise(X, noise, snr_db, rng):
    out = np.empty_like(X)
    for i, beat in enumerate(X):
        start = rng.integers(0, len(noise) - X.shape[1])
        segment = noise[start : start + X.shape[1]]
        scale = np.sqrt(np.mean(beat**2) / (np.mean(segment**2) * 10 ** (snr_db / 10)))
        out[i] = beat + scale * segment
    return out


def main():
    test_ds = BeatDataset("test")
    X, y = test_ds.X, test_ds.y

    model = ECGNet(channels=(8, 16, 32))
    model.load_state_dict(
        torch.load(PROJECT_ROOT / "results" / "model.pt", map_location="cpu")
    )
    model.eval()

    noise = wfdb.rdrecord(str(NSTDB / "em")).p_signal[:, 0]
    noise = (noise - noise.mean()) / noise.std()

    rng = np.random.default_rng(42)
    rows = []

    for label, X_eval in [
        ("clean", X),
        ("12 dB", add_noise(X, noise, 12, rng)),
        ("6 dB", add_noise(X, noise, 6, rng)),
        ("0 dB", add_noise(X, noise, 0, rng)),
    ]:
        with torch.no_grad():
            pred = model(torch.from_numpy(X_eval).unsqueeze(1)).argmax(1).numpy()

        acc = accuracy_score(y, pred)
        macro = f1_score(y, pred, average="macro", zero_division=0)
        per_class = f1_score(y, pred, average=None, zero_division=0)

        print(
            f"{label:<6} accuracy {acc:.4f}  macro F1 {macro:.4f}  "
            f"per-class {np.round(per_class, 3)}",
            flush=True,
        )

        row = {"condition": label, "accuracy": acc, "macro_f1": macro}
        row.update({f"f1_{c}": s for c, s in zip(CLASSES, per_class)})
        rows.append(row)

    out = PROJECT_ROOT / "results" / "noise_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "noise_results.csv"
    df.to_csv(out, index=False)

    x = range(4)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, df["accuracy"], "o-", label="accuracy")
    ax.plot(x, df["macro_f1"], "s-", label="macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(["clean", "12 dB", "6 dB", "0 dB"])
    ax.set_xlabel("SNR")
    ax.set_ylabel("score")
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3)
    ax.legend()


if __name__ == "__main__":
    main()
