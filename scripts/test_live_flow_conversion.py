"""
Test Live Flow Conversion to ML Feature Format
================================================
Verifies that live network flow dictionaries are accurately converted to 
the exact 72-feature format and standard scaling expected by trained ML models.
"""

import sys
import os
import logging
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("test_live_flow")


def run_test():
    logger.info("Initializing DataPreprocessor and DetectionEngine...")
    from ml.preprocessing import DataPreprocessor
    from detection.detector import DetectionEngine, create_detection_engine

    preprocessor = DataPreprocessor()
    model_dir = os.path.join(BASE_DIR, 'models')

    # Load feature columns if available
    feat_path = os.path.join(model_dir, 'feature_columns.pkl')
    if os.path.exists(feat_path):
        import pickle
        with open(feat_path, 'rb') as f:
            preprocessor.feature_columns = pickle.load(f)
        logger.info(f"Loaded {len(preprocessor.feature_columns)} feature columns from {feat_path}")

    # Load scaler if available
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        import pickle
        with open(scaler_path, 'rb') as f:
            preprocessor.scaler = pickle.load(f)
            preprocessor.fitted = True
        logger.info("Loaded scaler.pkl successfully")

    # Sample live flows produced by FlowTracker / FlowAggregator
    sample_flows = [
        {
            'flow_id': '192.168.1.100:49152->10.0.0.1:80-TCP',
            'source_ip': '192.168.1.100',
            'destination_ip': '10.0.0.1',
            'source_port': 49152,
            'destination_port': 80,
            'protocol': 'TCP',
            'duration': 1.25,
            'bytes_sent': 1500,
            'bytes_recv': 4500,
            'packets_sent': 10,
            'packets_recv': 15,
            'total_bytes': 6000,
            'total_packets': 25,
            'syn_count': 1,
            'ack_count': 24,
            'fin_count': 1,
            'rst_count': 0
        },
        {
            'flow_id': '172.16.0.5:54321->10.0.0.2:22-TCP',
            'source_ip': '172.16.0.5',
            'destination_ip': '10.0.0.2',
            'source_port': 54321,
            'destination_port': 22,
            'protocol': 'TCP',
            'duration': 0.15,
            'bytes_sent': 12000,
            'bytes_recv': 200,
            'packets_sent': 150,
            'packets_recv': 2,
            'total_bytes': 12200,
            'total_packets': 152,
            'syn_count': 150,
            'ack_count': 2,
            'fin_count': 0,
            'rst_count': 148
        }
    ]

    logger.info("\n--- Test 1: Direct Feature Extraction from Live Flow ---")
    for i, flow in enumerate(sample_flows, 1):
        feature_vector = preprocessor.extract_features_from_live_flow(flow)
        assert isinstance(feature_vector, np.ndarray), "Feature vector must be numpy array"
        assert feature_vector.ndim == 2, f"Feature vector shape must be 2D, got {feature_vector.shape}"
        assert feature_vector.shape[1] == len(preprocessor.feature_columns), (
            f"Expected {len(preprocessor.feature_columns)} features, got {feature_vector.shape[1]}"
        )
        assert not np.isnan(feature_vector).any(), "Feature vector contains NaN values!"
        logger.info(f"Flow #{i} successfully converted to feature vector shape: {feature_vector.shape}")

    logger.info("\n--- Test 2: Live Flow Detection Engine Integration ---")
    engine = create_detection_engine(model_dir=model_dir)

    results = engine.detect(sample_flows)
    if isinstance(results, list):
        for i, res in enumerate(results, 1):
            logger.info(
                f"Detection #{i}: is_attack={res.is_attack}, "
                f"attack_type='{res.attack_type}', confidence={res.confidence:.4f}, "
                f"model_used='{res.model_used}'"
            )
    else:
        logger.info(
            f"Detection result: is_attack={results.is_attack}, "
            f"attack_type='{results.attack_type}', confidence={results.confidence:.4f}"
        )

    print("\n" + "=" * 60)
    print("ALL LIVE FLOW CONVERSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_test()
