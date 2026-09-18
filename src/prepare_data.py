from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml
from scipy.signal import butter, sosfiltfilt

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    with open(PROJECT_ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


def load_record(record_id, raw_dir, lead):
    record = wfdb.rdrecord(str(raw_dir / record_id))
    names = [n.strip() for n in record.sig_name]
    if lead not in names:
        raise ValueError(f"record {record_id} has no lead {lead}: {names}")
    return record.p_signal[:, names.index(lead)], record.fs


def preprocess(signal, fs, cfg):
    p = cfg["preprocessing"]
    if p["bandpass"]["enabled"]:
        bp = p["bandpass"]
        sos = butter(
            bp["order"],
            [bp["low_hz"] / (fs / 2), bp["high_hz"] / (fs / 2)],
            btype="band",
            output="sos",
        )
        signal = sosfiltfilt(sos, signal)
    if p["normalization"]["method"] == "per_record_zscore":
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    if p["clipping"]["enabled"]:
        limit = p["clipping"]["limit_sigma"]
        signal = np.clip(signal, -limit, limit)
    return signal


def extract_beats(signal, record_id, raw_dir, cfg):
    ann = wfdb.rdann(str(raw_dir / record_id), "atr")
    before, after = cfg["beats"]["samples_before"], cfg["beats"]["samples_after"]
    aami, keep = cfg["labels"]["aami_map"], set(cfg["labels"]["classes"])

    windows, labels, dropped = [], [], 0
    for symbol, sample in zip(ann.symbol, ann.sample):
        if aami.get(symbol) not in keep:
            continue
        if sample - before < 0 or sample + after > len(signal):
            dropped += 1
            continue
        windows.append(signal[sample - before : sample + after])
        labels.append(aami[symbol])

    return np.asarray(windows, np.float32), np.asarray(labels), dropped


def main():
    cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    out_dir = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for split in ["train", "val", "test"]:
        X, y, src, dropped = [], [], [], 0
        print(f"\n{split}:")
        for record_id in cfg["records"][split]:
            signal, fs = load_record(record_id, raw_dir, cfg["dataset"]["lead"])
            signal = preprocess(signal, fs, cfg)
            w, lab, n = extract_beats(signal, record_id, raw_dir, cfg)
            X.append(w)
            y.append(lab)
            src.extend([record_id] * len(lab))
            dropped += n
            print(f"  {record_id}: {len(lab):>5} beats, {n} dropped")

        X, y, src = np.concatenate(X), np.concatenate(y), np.asarray(src)
        np.savez_compressed(out_dir / f"{split}.npz", X=X, y=y, record=src)

        row = {
            "split": split,
            "records": len(cfg["records"][split]),
            "dropped": dropped,
        }
        row.update({c: int((y == c).sum()) for c in cfg["labels"]["classes"]})
        row["total"] = len(y)
        stats.append(row)

    table = pd.DataFrame(stats)
    table.to_csv(PROJECT_ROOT / "results" / "dataset_stats.csv", index=False)
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
