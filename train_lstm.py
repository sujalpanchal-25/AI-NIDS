"""
LSTM Neural Network Retraining Script for AI-NIDS
=================================================
Trains the PyTorch LSTM sequence classification model on 3 datasets:
1. KDDTrain+.txt
2. KDDTest+.txt
3. UNSW_NB15_training-set.csv
"""

import os
import sys
import logging
import json
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-5s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("train_lstm")

# Base paths
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data" / "datasets"
MODEL_DIR     = BASE_DIR / "models"
SAVED_MODEL_DIR = BASE_DIR / "data" / "saved_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
KDD_TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
KDD_TEST_PATH  = DATA_DIR / "KDDTest+.txt"
UNSW_PATH      = DATA_DIR / "UNSW_NB15_training-set.csv"

# Model output paths
LSTM_MODEL_PATH       = MODEL_DIR / "lstm_model.pt"
LSTM_DETECTOR_PATH    = MODEL_DIR / "lstm_detector.pt"
SAVED_LSTM_PATH       = SAVED_MODEL_DIR / "lstm_detector.pt"
LSTM_SCALER_PATH      = MODEL_DIR / "lstm_scaler.pkl"
LSTM_FEATURES_PATH    = MODEL_DIR / "lstm_feature_columns.pkl"
LSTM_METADATA_PATH    = MODEL_DIR / "lstm_metadata.json"

KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


def load_unsw_dataset(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and preprocess UNSW_NB15 dataset."""
    logger.info(f"Loading UNSW-NB15 dataset: {path.name}")
    df = pd.read_csv(path)
    logger.info(f"  Loaded UNSW-NB15: {len(df):,} rows x {len(df.columns)} cols")

    y = df['label'].values
    drop_cols = ['id', 'label', 'attack_cat', 'proto', 'service', 'state']
    X_df = df.drop(columns=drop_cols, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number]).fillna(0)

    rename_map = {
        'dur': 'duration',
        'sbytes': 'bytes_sent',
        'dbytes': 'bytes_recv',
        'spkts': 'packets_sent',
        'dpkts': 'packets_recv',
        'sttl': 'src_ttl',
        'dttl': 'dst_ttl'
    }
    X_df = X_df.rename(columns=rename_map)
    return X_df, y


def load_kdd_dataset(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and preprocess NSL-KDD dataset."""
    logger.info(f"Loading NSL-KDD dataset: {path.name}")
    df = pd.read_csv(path, header=None, names=KDD_COLUMNS)
    logger.info(f"  Loaded NSL-KDD ({path.name}): {len(df):,} rows x {len(df.columns)} cols")

    benign = {'normal', 'benign', '0'}
    y = df['label'].astype(str).str.strip().str.lower().apply(lambda v: 0 if v in benign else 1).values

    drop_cols = ['label', 'difficulty', 'protocol_type', 'service', 'flag']
    X_df = df.drop(columns=drop_cols, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number]).fillna(0)

    rename_map = {
        'src_bytes': 'bytes_sent',
        'dst_bytes': 'bytes_recv',
        'count': 'packets_sent',
        'srv_count': 'packets_recv'
    }
    X_df = X_df.rename(columns=rename_map)
    return X_df, y


def combine_all_datasets() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Combine KDDTrain+, KDDTest+, and UNSW_NB15 into a single dataset."""
    frames = []
    labels = []
    datasets_used = []

    if KDD_TRAIN_PATH.exists():
        df_kdd_tr, y_kdd_tr = load_kdd_dataset(KDD_TRAIN_PATH)
        frames.append(df_kdd_tr)
        labels.append(y_kdd_tr)
        datasets_used.append("KDDTrain+.txt")

    if KDD_TEST_PATH.exists():
        df_kdd_te, y_kdd_te = load_kdd_dataset(KDD_TEST_PATH)
        frames.append(df_kdd_te)
        labels.append(y_kdd_te)
        datasets_used.append("KDDTest+.txt")

    if UNSW_PATH.exists():
        df_unsw, y_unsw = load_unsw_dataset(UNSW_PATH)
        frames.append(df_unsw)
        labels.append(y_unsw)
        datasets_used.append("UNSW_NB15_training-set.csv")

    if not frames:
        raise FileNotFoundError("No datasets found in data/datasets!")

    combined_df = pd.concat(frames, axis=0, ignore_index=True).fillna(0)
    combined_y = np.concatenate(labels, axis=0)

    logger.info(f"Combined Dataset Total Samples: {len(combined_df):,} across {len(combined_df.columns)} features.")
    return combined_df, combined_y, datasets_used


def main():
    print("=" * 60)
    print("  AI-NIDS  --  LSTM Neural Network Training Engine")
    print("=" * 60)

    # 1. Load data from all 3 datasets
    combined_df, y, datasets_used = combine_all_datasets()
    feature_columns = list(combined_df.columns)

    # 2. Scale features
    logger.info("Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_df.values)

    # 3. Initialize PyTorch LSTM detector
    from ml.models.lstm_detector import LSTMDetector
    
    sequence_length = 10
    logger.info(f"Initializing Bidirectional LSTM Detector (Sequence Length: {sequence_length})...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training on device: {device.upper()}")

    detector = LSTMDetector(
        input_dim=X_scaled.shape[1],
        sequence_length=sequence_length,
        hidden_dim=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        bidirectional=True,
        learning_rate=0.001,
        batch_size=256,
        epochs=12,
        device=device
    )

    # 4. Train-Test Split & Train LSTM Model
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)

    logger.info(f"Training set: {len(X_tr):,} samples | Validation set: {len(X_va):,} samples")
    logger.info("Starting LSTM Model Training...")
    history = detector.train(X_tr, y_tr, X_val=X_va, y_val=y_va, verbose=True)

    # 5. Evaluate Model
    logger.info("Evaluating trained LSTM Detector...")
    metrics = detector.evaluate(X_scaled[-10000:], y[-10000:])
    
    acc = metrics.get('accuracy', 0.0)
    f1  = metrics.get('f1', 0.0)
    prec = metrics.get('precision', 0.0)
    rec  = metrics.get('recall', 0.0)

    logger.info(f"LSTM Training Accuracy : {acc * 100:.2f}%")
    logger.info(f"LSTM F1-Score         : {f1 * 100:.2f}%")
    logger.info(f"LSTM Precision        : {prec * 100:.2f}%")
    logger.info(f"LSTM Recall           : {rec * 100:.2f}%")

    # 6. Save model binaries
    logger.info("Saving LSTM model artifacts...")
    detector.save(str(LSTM_MODEL_PATH))
    detector.save(str(LSTM_DETECTOR_PATH))
    detector.save(str(SAVED_LSTM_PATH))

    joblib.dump(scaler, LSTM_SCALER_PATH)
    joblib.dump(feature_columns, LSTM_FEATURES_PATH)

    metadata = {
        "model_type": "LSTM Neural Network",
        "trained_at": pd.Timestamp.now().isoformat(),
        "datasets_used": datasets_used,
        "total_samples": int(len(combined_df)),
        "input_dim": int(X_scaled.shape[1]),
        "sequence_length": sequence_length,
        "metrics": {
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4)
        }
    }

    with open(LSTM_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print("  LSTM Retraining Complete!")
    print(f"  Datasets Used: {', '.join(datasets_used)}")
    print(f"  Total Samples: {len(combined_df):,}")
    print(f"  Accuracy     : {acc * 100:.2f}%")
    print(f"  Saved Files  : {LSTM_MODEL_PATH.name}, {SAVED_LSTM_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
