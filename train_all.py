"""
AI-NIDS Master Retraining Orchestrator
=======================================
Runs all model training scripts in proper sequential dependency order:
  1. XGBoost Classifier (train.py)
  2. LSTM Neural Network (train_lstm.py)
  3. Autoencoder Anomaly Detector (train_autoencoder.py)
  4. Adaptive Dynamic Ensemble (train_adaptive_ensemble.py)

Usage:
    python train_all.py
"""

import sys
import time
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-5s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("train_all")

BASE_DIR = Path(__file__).parent
venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
if venv_python.exists():
    PYTHON_EXE = str(venv_python)
else:
    PYTHON_EXE = sys.executable

TRAINING_SCRIPTS = [
    ("XGBoost Classifier", "train.py"),
    ("LSTM Neural Network", "train_lstm.py"),
    ("Anomaly Autoencoder", "train_autoencoder.py"),
    ("Graph Neural Network (GNN)", "train_gnn.py"),
    ("Multi-Window Temporal Detector", "train_temporal.py"),
    ("Adaptive Dynamic Ensemble", "train_adaptive_ensemble.py")
]


def run_script(name: str, script_name: str) -> bool:
    """Run a single training script and report status."""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    print("\n" + "=" * 70)
    print(f" 🚀 Executing [{name}] -> {script_name}")
    print("=" * 70)

    start_time = time.time()
    try:
        res = subprocess.run(
            [PYTHON_EXE, str(script_path)],
            cwd=str(BASE_DIR),
            check=True
        )
        elapsed = time.time() - start_time
        logger.info(f"✅ [{name}] completed successfully in {elapsed:.2f} seconds.")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ [{name}] failed with returncode {e.returncode} after {elapsed:.2f} seconds.")
        return False
    except Exception as e:
        logger.error(f"❌ [{name}] encountered unexpected error: {e}")
        return False


def main():
    print("=" * 70)
    print(" 🛡️  AI-NIDS MASTER MODEL TRAINING PIPELINE")
    print("=" * 70)

    results = {}
    overall_start = time.time()

    for name, script in TRAINING_SCRIPTS:
        success = run_script(name, script)
        results[name] = success
        if not success:
            logger.warning(f"⚠️ Pipeline step '{name}' failed. Continuing to next step...")

    overall_elapsed = time.time() - overall_start

    print("\n" + "=" * 70)
    print(" 📊 TRAINING PIPELINE SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        status_str = "✅ SUCCESS" if ok else "❌ FAILED"
        print(f"  - {name:30s} : {status_str}")

    print("-" * 70)
    total_passed = sum(1 for ok in results.values() if ok)
    print(f"  Total Steps Completed: {total_passed}/{len(TRAINING_SCRIPTS)}")
    print(f"  Total Elapsed Time   : {overall_elapsed / 60:.2f} minutes")
    print("=" * 70 + "\n")

    if total_passed == len(TRAINING_SCRIPTS):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
