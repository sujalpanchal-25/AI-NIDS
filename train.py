"""
AI-NIDS Model Training Script
==============================
Run this ONCE to train and save ML models.
After this, real ML detection runs instead of heuristic rules.

Usage:
    python train.py
"""

import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ── Setup logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("train")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATASET       = BASE_DIR / "data" / "datasets" / "sample_traffic.csv"
MODEL_DIR     = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

XGBOOST_PATH  = MODEL_DIR / "xgboost_model.pkl"
SCALER_PATH   = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"
METADATA_PATH = MODEL_DIR / "training_metadata.json"


# ==============================================================================
# 1. LOAD
# ==============================================================================
def load_dataset(path: Path) -> pd.DataFrame:
    logger.info(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df


# ==============================================================================
# 2. PREPROCESS
# ==============================================================================
def preprocess(df: pd.DataFrame):
    logger.info("Preprocessing ...")

    # Find label column
    label_col = None
    for c in ["label", "Label", "class", "Class", "attack_type",
              "attack", "category", "Category", "type"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(
            "No label column found! Expected: label / Label / class / attack_type"
        )
    logger.info(f"Label column: {label_col!r}")
    logger.info(f"Label distribution:\n{df[label_col].value_counts()}\n")

    # Drop non-feature columns
    drop_cols = [label_col]
    for col in ["timestamp", "Timestamp", "src_ip", "dst_ip",
                "Source IP", "Destination IP", "flow_id", "Flow ID"]:
        if col in df.columns:
            drop_cols.append(col)

    X_raw = df.drop(columns=drop_cols, errors="ignore")
    X = X_raw.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError("No numeric columns found! Check your CSV.")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    feature_columns = list(X.columns)
    logger.info(f"Numeric features ({len(feature_columns)}): {feature_columns}")

    # Binary encode: BENIGN/NORMAL → 0, anything else → 1
    benign = {"BENIGN", "NORMAL", "LEGITIMATE", "CLEAN", "0"}
    y = (
        df[label_col]
        .astype(str).str.strip().str.upper()
        .apply(lambda v: 0 if v in benign else 1)
        .values
    )
    logger.info(f"Normal (0): {(y==0).sum():,}  |  Attack (1): {(y==1).sum():,}")
    return X.values, y, feature_columns


# ==============================================================================
# 3. TRAIN
# ==============================================================================
def train_model(X_train, y_train, X_test, y_test):
    try:
        import xgboost as xgb
        logger.info("Training XGBoost ...")
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw   = n_neg / max(n_pos, 1)
        logger.info(f"scale_pos_weight = {spw:.2f}")

        model = xgb.XGBClassifier(
            n_estimators     = 200,
            max_depth        = 6,
            learning_rate    = 0.1,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            scale_pos_weight = spw,
            objective        = "binary:logistic",
            eval_metric      = "auc",
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

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

    y_pred  = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"\nAccuracy: {accuracy*100:.2f}%")
    logger.info("\n" + classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"]
    ))
    return model, accuracy


# ==============================================================================
# 4. SAVE
# ==============================================================================
def save_all(model, scaler, feature_columns, accuracy):
    import json

    with open(XGBOOST_PATH,  "wb") as f: pickle.dump(model,           f)
    with open(SCALER_PATH,   "wb") as f: pickle.dump(scaler,          f)
    with open(FEATURES_PATH, "wb") as f: pickle.dump(feature_columns, f)

    meta = {
        "trained_at":      datetime.utcnow().isoformat(),
        "model_type":      type(model).__name__,
        "accuracy":        round(accuracy, 4),
        "feature_count":   len(feature_columns),
        "feature_columns": feature_columns,
        "dataset":         str(DATASET),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  xgboost_model.pkl  -> {XGBOOST_PATH}")
    logger.info(f"  scaler.pkl         -> {SCALER_PATH}")
    logger.info(f"  feature_columns.pkl-> {FEATURES_PATH}")
    logger.info(f"  training_metadata  -> {METADATA_PATH}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "=" * 55)
    print("  AI-NIDS  --  Model Training Script")
    print("=" * 55 + "\n")

    if not DATASET.exists():
        logger.error(f"Dataset not found: {DATASET}")
        sys.exit(1)

    df = load_dataset(DATASET)
    X, y, feature_columns = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    scaler   = StandardScaler()
    X_trains = scaler.fit_transform(X_train)
    X_tests  = scaler.transform(X_test)

    model, accuracy = train_model(X_trains, y_train, X_tests, y_test)

    logger.info("Saving model files ...")
    save_all(model, scaler, feature_columns, accuracy)

    print("\n" + "=" * 55)
    print(f"  Training Complete!")
    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Files    : models/")
    print()
    print("  Next steps:")
    print("  1. Restart server  ->  python run.py")
    print("  2. Upload any CSV  ->  Real ML runs!")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
