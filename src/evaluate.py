import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from pathlib import Path
from model import ECGNet, count_parameters


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--data", default="data/processed/test.npz")
args = parser.parse_args()

model_name = Path(args.model).stem

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

report = classification_report(
    y,
    pred,
    labels=[0, 1, 2, 3],
    target_names=classes,
    zero_division=0
)

matrix = pd.DataFrame(
    confusion_matrix(y, pred, labels=[0, 1, 2, 3]),
    index=classes,
    columns=classes
)

output = Path(f"results/evaluation_{model_name}.txt")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", encoding="utf-8") as f:
    f.write(f"Model: {model_name}\n")
    f.write(f"Dataset: {args.data}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Parameters: {count_parameters(model)}\n")

    f.write("\nPer-class results:\n")
    f.write(report)

    f.write("\nConfusion matrix:\n")
    f.write(matrix.to_string())

print(f"Results saved to {output}")