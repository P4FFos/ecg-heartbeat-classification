import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.model import ECGNet, count_parameters


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--data", default="data/processed/test.npz")
args = parser.parse_args()

classes = ["N", "S", "V", "F"]

# Load data

data = np.load(args.data)

X = data["X"]
y = data["y"]

y = np.array([classes.index(label) for label in y])

if X.ndim == 2:
    X = X[:, None, :]

X = torch.tensor(X, dtype=torch.float32)

# Load model
model = ECGNet(channels=(8, 16, 32))
model.load_state_dict(torch.load(args.model, map_location="cpu"))
model.eval()


# Predict
with torch.no_grad():
    pred = model(X).argmax(1).numpy()


# Results

accuracy = accuracy_score(y, pred)
macro_f1 = f1_score(y, pred, average="macro", zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")

print("\nPer-class results:")
print(classification_report(
    y,
    pred,
    labels=[0, 1, 2, 3],
    target_names=classes,
    zero_division=0
))

print("Confusion matrix:")
print(pd.DataFrame(
    confusion_matrix(y, pred, labels=[0, 1, 2, 3]),
    index=classes,
    columns=classes
))

print(f"\n{count_parameters(model)} parameters")