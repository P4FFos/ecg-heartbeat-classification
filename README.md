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

# Run order 
1. Prepare the data
2. Train the model
3. Evaluation on the test set
4. Noise test
5. Error analysis

# Train
Trains a CNN (`ECGNet`, channels 8/16/32, 6612 parameters)

Output: `results/model.pt` (the trained weights)

# Evaluate on the test set
Runs the model on the test set and prints accuracy, macro F1, per-class results, and the confusion matrix

# Noise test 
Adds noise to the test beats at different strengths and checks how much accuracy drops. Noise is added only when testing, never during training.

Output: `results/noise_results.csv` and `results/noise_plot.png`

# Error analysis
Plots examples of beats the model got wrong, so the errors can be looked at directly instead of only as numbers.

# Quantization 
```
python -m src.quantize
```

Output:
- `results/quantize_results.csv` (metrics per variant)

# MAX78002 Setup
Deployment uses the official Analog Devices tools for MAX78000/MAX78002. Clone both in the same directory as repo, not inside it:
```
cd ..
git clone --recursive https://github.com/analogdevicesinc/ai8x-training.git
git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git
```

Each tool needs its own Python 3.11 environment. Before running pip change:
- `ai8x-training/requirements-datasets.txt`: `pyffmpeg==2.4.2.18.1` -> `pyffmpeg==2.5.2.3.2`
- `ai8x-synthesis/requirements.txt` (Windows only): `matplotlib==3.9.1` -> `matplotlib==3.9.1.post1`

Mac/Linux:
```
cd ai8x-training
python3.11 -m venv .venv
source .venv/bin/activate
pip install "setuptools<81"
pip install visdom --no-build-isolation
pip install -r requirements.txt
deactivate

cd ../ai8x-synthesis
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```
cd ai8x-training
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install "setuptools<81"
pip install visdom --no-build-isolation
pip install -r requirements.txt
deactivate

cd ..\ai8x-synthesis
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

To build the C code, install the MSDK: https://analogdevicesinc.github.io/msdk/

# MAX78002 Training
Run `python src/prepare_data.py`

Copy the model and the dataset loader into `ai8x-training`:
- `max78002/model_configuration/ecgnet.py` -> `ai8x-training/models/`
- `max78002/model_configuration/mitbih.py` -> `ai8x-training/datasets/`

From `ai8x-training`:
```
python train.py --model ecgnet --dataset MITBIH --data ../Group19_ESG_Project/data/processed --device MAX78002 --use-bias --epochs 60 --optimizer Adam --lr 0.001 --batch-size 128 --deterministic --seed 42 --workers 0 --confusion --name ecgnet
```

Output: `logs/ecgnet___<date>/ecgnet_qat_best.pth.tar`, copy it to `max78002/quantization_output/ecgnet_qat.pth.tar`

# MAX78002 quantization
From `ai8x-synthesis`:
```
python quantize.py ../Group19_ESG_Project/max78002/quantization_output/ecgnet_qat.pth.tar ../Group19_ESG_Project/max78002/quantization_output/ecgnet_q8.pth.tar --device MAX78002
```

Output: `max78002/quantization_output/ecgnet_q8.pth.tar` (8-bit weights)

Evaluate the 8-bit model on the test set, from `ai8x-training`:
```
python train.py --model ecgnet --dataset MITBIH --data ../Group19_ESG_Project/data/processed --device MAX78002 --use-bias --evaluate --exp-load-weights-from ../Group19_ESG_Project/max78002/quantization_output/ecgnet_q8.pth.tar -8 --confusion --workers 0 --deterministic --seed 42 --save-sample 10
```

Output: `sample_mitbih.npy` (one test beat for the C code self-test), move it to `max78002/quantization_output/`

# MAX78002 C code
From `ai8x-synthesis`:
```
python ai8xize.py --test-dir synthed_net --prefix ecgnet --checkpoint-file ../Group19_ESG_Project/max78002/quantization_output/ecgnet_q8.pth.tar --config-file ../Group19_ESG_Project/max78002/model_configuration/ecgnet.yaml --sample-input ../Group19_ESG_Project/max78002/quantization_output/sample_mitbih.npy --device MAX78002 --timer 0 --display-checkpoint --verbose
```

`max78002/model_configuration/ecgnet.yaml` describes how the layers are placed on the CNN accelerator.

Output: `synthed_net/ecgnet/`. Copy `main.c`, `cnn.c`, `cnn.h`, `weights.h`, `sampledata.h`, `sampleoutput.h`, `softmax.c`, `Makefile` and `project.mk` to `max78002/device_files/`, and `log.txt` (synthesis report) to `max78002/synthesis_output/`

Build from a terminal:

Mac/Linux:
```
source ~/MaximSDK/setenv.sh
cd max78002/device_files
make MAXIM_PATH=$HOME/MaximSDK
```

Windows (PowerShell, not Git Bash):
```
$env:PATH = "C:\MaximSDK\Tools\GNUTools\10.3\bin;C:\MaximSDK\Tools\MSYS2\usr\bin;$env:PATH"
cd max78002/device_files
make MAXIM_PATH=C:/MaximSDK
```

On the actual board, the program should run the sample beat on the CNN accelerator and prints `PASS` if the output matches the expected one
