# Edge-Ready ECG Classification using CRISP-DM

Classifies MIT-BIH heartbeats into 4 classes N, S, V, F. 

# Setup 

Mac:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows: 
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

# Download the data

Download the MIT-BIH Database from PhysioNet:
https://physionet.org/content/mitdb/1.0.0/

Click "Download the ZIP file" and unzip it into `data/`, you will get: 

data/mit-bih-arrhythmia-database-1.0.0/100.dat, 100.hea, 100.atr, ...

# Run the EDA

Open `notebooks/EDA.ipynb` and run all cells. It will write these tables to `results/`:
`record_metadata.csv`, `symbol_counts.csv`, `class_counts.csv`, `signal_quality.csv`

# Prepare the data
```
python src/prepare_data.py
```

All settings are in the `configs/data.yaml`

The script does: 
MLII lead -> band-pass filder 0.5-40 Hz -> z-score per record -> clip -> 200 sample beat windows

Output:
- `data/processed/train.npz`, `val.npz`, `test.npz` (arrays `X`, `y`, `record`)
- `results/dataset_stats.csv` (beats per class in each split)

# Data split
- Train: 101, 106, 108, 109, 112, 115, 116, 118, 122, 201, 203, 207, 208, 209, 215, 230
- Validation: 114, 119, 124, 205, 220, 223
- Test: 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234
- Excluded: 102, 104, 107, 217

# Quantization 
```
python -m src.quantize
```

Output:
- `results/quantize_results.csv` (metrics per variant)
