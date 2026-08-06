"""
AI-NIDS Model Training Script (Multi-Dataset Retraining)
=========================================================
Retrains XGBoost classifier using:
  1. KDDTest+.txt (NSL-KDD Dataset)
  2. UNSW_NB15_training-set.csv (UNSW-NB15 Dataset)

Usage:
    python train.py
"""

import os
import sys
import pickle
import logging
import json
from typing import Tuple, List
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# ── Setup logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data" / "datasets"
MODEL_DIR     = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

KDD_PATH      = DATA_DIR / "KDDTrain+.txt"
KDD_TEST_PATH = DATA_DIR / "KDDTest+.txt"
UNSW_PATH     = DATA_DIR / "UNSW_NB15_training-set.csv"

XGBOOST_PATH  = MODEL_DIR / "xgboost_model.pkl"
SCALER_PATH   = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"
METADATA_PATH = MODEL_DIR / "training_metadata.json"

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


def remove_old_models():
    """Remove existing trained model files before retraining."""
    logger.info("Cleaning up old model files...")
    for file_path in [XGBOOST_PATH, SCALER_PATH, FEATURES_PATH, METADATA_PATH]:
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"  Deleted old model file: {file_path.name}")
            except Exception as e:
                logger.warning(f"  Could not delete {file_path.name}: {e}")


def load_unsw_dataset(path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and preprocess UNSW_NB15 dataset."""
    logger.info(f"Loading UNSW-NB15 dataset: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded UNSW-NB15: {len(df):,} rows x {len(df.columns)} cols")

    # Binary label: 0 for Normal, 1 for Attack
    y = df['label'].values

    # Drop non-feature and text columns
    drop_cols = ['id', 'label', 'attack_cat', 'proto', 'service', 'state']
    X_df = df.drop(columns=drop_cols, errors='ignore')
    X_df = X_df.select_dtypes(include=[np.number]).fillna(0)

    # Standardize column names to common network features
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
    """Load and preprocess NSL-KDD dataset (KDDTrain+.txt)."""
    logger.info(f"Loading NSL-KDD dataset: {path.name}")
    df = pd.read_csv(path, header=None, names=KDD_COLUMNS)
    logger.info(f"Loaded NSL-KDD ({path.name}): {len(df):,} rows x {len(df.columns)} cols")

    # Binary label: 0 for normal, 1 for any attack
    benign = {'normal', 'benign', '0'}
    y = df['label'].astype(str).str.strip().str.lower().apply(lambda v: 0 if v in benign else 1).values

    # Select numeric features only
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


def combine_datasets() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Combine UNSW-NB15 and KDDTrain+ datasets into a unified feature set."""
    datasets_used = []
    frames = []
    labels = []

    if UNSW_PATH.exists():
        df_unsw, y_unsw = load_unsw_dataset(UNSW_PATH)
        frames.append(df_unsw)
        labels.append(y_unsw)
        datasets_used.append("UNSW_NB15_training-set.csv")

    if KDD_PATH.exists():
        df_kdd, y_kdd = load_kdd_dataset(KDD_PATH)
        frames.append(df_kdd)
        labels.append(y_kdd)
        datasets_used.append("KDDTrain+.txt")
    elif KDD_TEST_PATH.exists():
        df_kdd, y_kdd = load_kdd_dataset(KDD_TEST_PATH)
        frames.append(df_kdd)
        labels.append(y_kdd)
        datasets_used.append("KDDTest+.txt")

    if not frames:
        raise FileNotFoundError("Neither UNSW_NB15_training-set.csv nor KDDTest+.txt were found!")

    # Concatenate dataframes with alignment on common feature names
    combined_df = pd.concat(frames, axis=0, ignore_index=True).fillna(0)
    combined_y = np.concatenate(labels, axis=0)

    feature_columns = list(combined_df.columns)
    logger.info(f"Combined Dataset Total Rows: {len(combined_df):,}")
    logger.info(f"Normal (0): {(combined_y == 0).sum():,}  |  Attack (1): {(combined_y == 1).sum():,}")
    logger.info(f"Combined Features ({len(feature_columns)}): {feature_columns}")

    return combined_df.values, combined_y, feature_columns, datasets_used


def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier with balanced class weights."""
    try:
        import xgboost as xgb
        logger.info("Training XGBoost Classifier...")
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = n_neg / max(n_pos, 1)
        logger.info(f"scale_pos_weight = {spw:.2f}")

        model = xgb.XGBClassifier(
            n_estimators     = 250,
            max_depth        = 8,
            learning_rate    = 0.08,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            scale_pos_weight = spw,
            objective        = "binary:logistic",
            eval_metric      = "auc",
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

    except ImportError:
        logger.warning("xgboost not installed — using RandomForest fallback")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators = 150,
            max_depth    = 12,
            class_weight = "balanced",
            random_state = 42,
            n_jobs       = -1,
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    logger.info(f"\nAccuracy : {accuracy*100:.2f}%")
    logger.info(f"Precision: {prec*100:.2f}%")
    logger.info(f"Recall   : {rec*100:.2f}%")
    logger.info(f"F1 Score : {f1*100:.2f}%\n")
    logger.info("\n" + classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"]
    ))
    return model, accuracy, prec, rec, f1


def save_all(model, scaler, feature_columns, accuracy, prec, rec, f1, datasets_used):
    """Save trained model, scaler, feature columns, and training metadata."""
    with open(XGBOOST_PATH,  "wb") as f: pickle.dump(model,           f)
    with open(SCALER_PATH,   "wb") as f: pickle.dump(scaler,          f)
    with open(FEATURES_PATH, "wb") as f: pickle.dump(feature_columns, f)

    meta = {
        "trained_at":      datetime.utcnow().isoformat(),
        "model_type":      type(model).__name__,
        "accuracy":        round(accuracy, 4),
        "precision":       round(prec, 4),
        "recall":          round(rec, 4),
        "f1_score":        round(f1, 4),
        "feature_count":   len(feature_columns),
        "feature_columns": feature_columns,
        "datasets":        datasets_used,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  xgboost_model.pkl  -> {XGBOOST_PATH}")
    logger.info(f"  scaler.pkl         -> {SCALER_PATH}")
    logger.info(f"  feature_columns.pkl-> {FEATURES_PATH}")
    logger.info(f"  training_metadata  -> {METADATA_PATH}")


def main():
    print("\n" + "=" * 60)
    print("  AI-NIDS  --  Multi-Dataset XGBoost Model Training")
    print("=" * 60 + "\n")

    # Step 1: Remove old model files before retraining
    remove_old_models()

    # Step 2: Combine datasets (KDDTest+.txt & UNSW_NB15_training-set.csv)
    X, y, feature_columns, datasets_used = combine_datasets()

    # Step 3: Train-Test Split (80% train / 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")

    # Step 4: Scale features
    scaler   = StandardScaler()
    X_trains = scaler.fit_transform(X_train)
    X_tests  = scaler.transform(X_test)

    # Step 5: Train Model
    model, accuracy, prec, rec, f1 = train_model(X_trains, y_train, X_tests, y_test)

    # Step 6: Save Model Artifacts
    logger.info("Saving updated model artifacts...")
    save_all(model, scaler, feature_columns, accuracy, prec, rec, f1, datasets_used)

    print("\n" + "=" * 60)
    print(f"  XGBoost Retraining Complete!")
    print(f"  Datasets Used: {', '.join(datasets_used)}")
    print(f"  Accuracy     : {accuracy * 100:.2f}%")
    print(f"  F1 Score     : {f1 * 100:.2f}%")
    print(f"  Features     : {len(feature_columns)}")
    print(f"  Artifacts    : models/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
