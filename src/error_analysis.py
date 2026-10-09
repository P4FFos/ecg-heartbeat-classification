from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset import BeatDataset
from src.model import ECGNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot(X, idx, title, ax, n=15):
    chosen = idx[:n]
    for i in chosen:
        ax.plot(X[i], color="black", alpha=0.25, linewidth=0.8)
    if len(chosen):
        ax.plot(X[chosen].mean(0), color="firebrick", linewidth=1.8)
    ax.axvline(100, color="steelblue", linestyle="--", linewidth=0.8)
    ax.set_title(f"{title}  (n={len(idx)})", fontsize=10)
    ax.set_xlabel("samples")


def main():
    test_ds = BeatDataset("test")
    X, y = test_ds.X, test_ds.y

    model = ECGNet(channels=(8, 16, 32))
    model.load_state_dict(
        torch.load(PROJECT_ROOT / "results" / "model.pt", map_location="cpu")
    )
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).unsqueeze(1)).argmax(1).numpy()

    cases = [
        ("N classified as S", 0, 1),
        ("S classified as N", 1, 0),
        ("F classified as S", 3, 1),
        ("F classified as V", 3, 2),
        ("N correct", 0, 0),
        ("V correct", 2, 2),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, (title, t, p) in zip(axes.flat, cases):
        plot(X, np.where((y == t) & (pred == p))[0], title, ax)

    plt.tight_layout()
    out = PROJECT_ROOT / "results" / "error_analysis.png"
    plt.savefig(out, dpi=150)
    print(f"saved to {out}")

    for title, t, p in cases:
        print(f"{title:<22} {np.sum((y == t) & (pred == p)):>6}")


if __name__ == "__main__":
    main()
