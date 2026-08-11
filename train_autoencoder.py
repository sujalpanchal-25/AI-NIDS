"""
Autoencoder Anomaly Detector Training Script for AI-NIDS
=========================================================
Loads KDD + UNSW datasets, extracts BENIGN/NORMAL network flows,
trains an AnomalyAutoencoder via reconstruction error minimization,
calibrates anomaly threshold on attack samples, and saves model artifacts.

Usage:
    python train_autoencoder.py
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ── Setup Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-5s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("train_autoencoder")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data" / "datasets"
MODEL_DIR       = BASE_DIR / "models"
SAVED_MODEL_DIR = BASE_DIR / "data" / "saved_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

KDD_TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
KDD_TEST_PATH  = DATA_DIR / "KDDTest+.txt"
UNSW_PATH      = DATA_DIR / "UNSW_NB15_training-set.csv"

AE_MODEL_PATH       = MODEL_DIR / "autoencoder_model.pt"
SAVED_AE_PATH       = SAVED_MODEL_DIR / "autoencoder_model.pt"
AE_METADATA_PATH    = MODEL_DIR / "autoencoder_metadata.json"
AE_SCALER_PATH      = MODEL_DIR / "autoencoder_scaler.pkl"

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
    """Load UNSW-NB15 dataset."""
    logger.info(f"Loading UNSW-NB15 dataset: {path.name}")
    df = pd.read_csv(path)
    y = df['label'].values.astype(int)
    drop_cols = ['id', 'label', 'attack_cat', 'proto', 'service', 'state']
    X_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)

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
    """Load NSL-KDD dataset."""
    logger.info(f"Loading NSL-KDD dataset: {path.name}")
    df = pd.read_csv(path, header=None, names=KDD_COLUMNS)
    benign = {'normal', 'benign', '0'}
    y = df['label'].astype(str).str.strip().str.lower().apply(
        lambda v: 0 if v in benign else 1
    ).values.astype(int)

    drop_cols = ['label', 'difficulty', 'protocol_type', 'service', 'flag']
    X_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).fillna(0)

    rename_map = {
        'src_bytes': 'bytes_sent',
        'dst_bytes': 'bytes_recv',
        'count': 'packets_sent',
        'srv_count': 'packets_recv'
    }
    X_df = X_df.rename(columns=rename_map)
    return X_df, y


def combine_datasets() -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    """Load and combine UNSW and KDD datasets into unified arrays."""
    frames, labels, datasets_used = [], [], []

    for loader_fn, path in [(load_unsw_dataset, UNSW_PATH),
                             (load_kdd_dataset, KDD_TRAIN_PATH),
                             (load_kdd_dataset, KDD_TEST_PATH)]:
        if not path.exists():
            continue
        X_df, y = loader_fn(path)
        frames.append(X_df)
        labels.append(y)
        datasets_used.append(path.name)

    if not frames:
        raise FileNotFoundError("No datasets found in data/datasets!")

    combined_df = pd.concat(frames, axis=0, ignore_index=True).fillna(0)
    X_all = combined_df.values.astype(np.float32)
    y_all = np.concatenate(labels)
    feature_count = X_all.shape[1]
    logger.info(f"Combined Dataset: {len(X_all):,} samples across {feature_count} features.")
    logger.info(f"  Normal (Benign): {(y_all == 0).sum():,} | Attack (Anomaly): {(y_all == 1).sum():,}")
    return X_all, y_all, feature_count, datasets_used


def main():
    print("=" * 60)
    print("  AI-NIDS  --  Autoencoder Anomaly Detector Training")
    print("=" * 60)

    # 1. Load combined dataset
    X_all, y_all, feat_dim, datasets_used = combine_datasets()

    # 2. Separate Benign (Normal) and Attack data
    X_normal = X_all[y_all == 0]
    X_attack = X_all[y_all == 1]

    # 3. Fit feature scaler on Benign data
    logger.info(f"Normalizing features with StandardScaler on {len(X_normal):,} benign samples...")
    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal).astype(np.float32)
    X_attack_scaled = scaler.transform(X_attack).astype(np.float32) if len(X_attack) > 0 else np.zeros((0, feat_dim), dtype=np.float32)

    # 4. Train/Validation split of normal traffic
    if len(X_normal_scaled) > 2000:
        X_train_norm, X_val_norm = train_test_split(X_normal_scaled, test_size=0.15, random_state=42)
    else:
        X_train_norm, X_val_norm = X_normal_scaled, None

    # 5. Initialize AnomalyAutoencoder
    from ml.models.autoencoder import AnomalyAutoencoder

    enc_dims = [128, 64, 32] if feat_dim >= 64 else ([64, 32, 16] if feat_dim >= 32 else [32, 16, 8])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f"Initializing AnomalyAutoencoder (Device: {device.upper()}, Hidden Dims: {enc_dims})...")
    ae = AnomalyAutoencoder(
        input_dim=feat_dim,
        encoding_dims=enc_dims,
        dropout_rate=0.15,
        learning_rate=0.001,
        batch_size=512,
        epochs=50,
        threshold_percentile=95.0,
        device=device
    )

    # 6. Train model on Normal traffic only
    logger.info(f"Training Autoencoder on {len(X_train_norm):,} BENIGN traffic samples...")
    history = ae.train(X_train_norm, X_val=X_val_norm, verbose=True)

    # 7. Calibrate decision threshold using Attack traffic
    if len(X_attack_scaled) > 0:
        err_norm = ae.get_reconstruction_error(X_val_norm if X_val_norm is not None else X_train_norm[:5000])
        err_atk  = ae.get_reconstruction_error(X_attack_scaled[:5000])

        logger.info(f"Normal Reconstruction Error Mean: {float(np.mean(err_norm)):.6f} (+/- {float(np.std(err_norm)):.6f})")
        logger.info(f"Attack Reconstruction Error Mean: {float(np.mean(err_atk)):.6f} (+/- {float(np.std(err_atk)):.6f})")

        # Grid search threshold to optimize F1 score on validation + attack mix
        val_X = np.vstack([X_val_norm if X_val_norm is not None else X_train_norm[:2000], X_attack_scaled[:2000]])
        val_y = np.concatenate([np.zeros(len(X_val_norm if X_val_norm is not None else X_train_norm[:2000]), dtype=int),
                                np.ones(len(X_attack_scaled[:2000]), dtype=int)])

        val_errors = ae.get_reconstruction_error(val_X)
        best_f1, best_t = 0.0, ae.threshold

        for pct in np.linspace(80, 99.5, 40):
            t_cand = float(np.percentile(err_norm, pct))
            preds = (val_errors > t_cand).astype(int)
            score = f1_score(val_y, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t_cand

        ae.threshold = float(best_t)
        logger.info(f"Calibrated Optimal Anomaly Threshold: {ae.threshold:.6f} (Validation F1: {best_f1 * 100:.2f}%)")

    # 8. Save Autoencoder Model Artifacts
    logger.info("Saving Autoencoder model artifacts...")
    ae.save(str(AE_MODEL_PATH))
    ae.save(str(SAVED_AE_PATH))

    with open(AE_SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    metadata = {
        "model_type": "AnomalyAutoencoder",
        "trained_at": pd.Timestamp.now().isoformat(),
        "datasets_used": datasets_used,
        "normal_samples_trained": len(X_normal),
        "attack_samples_evaluated": len(X_attack),
        "input_dim": feat_dim,
        "threshold": ae.threshold,
        "history": history
    }

    with open(AE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print("  Autoencoder Training Complete!")
    print(f"  Normal Samples Trained : {len(X_normal):,}")
    print(f"  Calibrated Threshold   : {ae.threshold:.6f}")
    print(f"  Saved Files            : {AE_MODEL_PATH.name}, {SAVED_AE_PATH.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
