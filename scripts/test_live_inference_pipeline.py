"""
Test Live Traffic ML Inference Pipeline Integration
====================================================
Tests end-to-end live flow detection through DetectionEngine, verifying:
  1. Benign live traffic evaluation
  2. Suspicious / high-volume traffic evaluation
  3. Normalized output fields: (prediction, attack_type, confidence, severity, risk_score)
  4. Integration with LiveCaptureManager flow aggregator
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_live_inference")


def run_pipeline_test():
    logger.info("=== Starting Live ML Inference Pipeline Test ===")
    from collectors.live_capture import LiveCaptureManager
    from detection.detector import create_detection_engine

    model_dir = os.path.join(BASE_DIR, 'models')
    detector = create_detection_engine(model_dir=model_dir)
    manager = LiveCaptureManager(detector=detector, model_dir=model_dir)

    # ---------------------------------------------------------
    # Test 1: Benign Live Traffic Test
    # ---------------------------------------------------------
    logger.info("\n[Step 1] Testing Benign Live Network Traffic...")
    benign_flow = {
        'flow_id': '192.168.1.105:51234->10.0.0.5:443-TCP',
        'source_ip': '192.168.1.105',
        'destination_ip': '10.0.0.5',
        'source_port': 51234,
        'destination_port': 443,
        'protocol': 'TCP',
        'duration': 2.5,
        'bytes_sent': 1200,
        'bytes_recv': 4800,
        'packets_sent': 8,
        'packets_recv': 12,
        'total_bytes': 6000,
        'total_packets': 20,
        'syn_count': 1,
        'ack_count': 19,
        'fin_count': 1,
        'rst_count': 0
    }

    # Process benign flow through manager
    manager._process_completed_flow(benign_flow)
    assert 'detection' in benign_flow, "Detection result missing from completed flow dictionary!"
    benign_result = benign_flow['detection']

    logger.info(f"Benign Flow Detection Output: {benign_result}")

    # Verify Normalized Output Fields
    required_fields = ['prediction', 'attack_type', 'confidence', 'severity', 'risk_score']
    for field in required_fields:
        assert field in benign_result, f"Required normalized field '{field}' missing from output!"

    assert benign_result['prediction'] == 0, f"Benign traffic falsely classified as attack! Got {benign_result}"
    assert benign_result['severity'] in ['INFO', 'LOW'], f"Unexpected severity for benign traffic: {benign_result['severity']}"

    # ---------------------------------------------------------
    # Test 2: Suspicious / High-Volume Traffic Test
    # ---------------------------------------------------------
    logger.info("\n[Step 2] Testing Suspicious High-Volume Flood Traffic...")
    suspicious_flow = {
        'flow_id': '172.16.0.99:65432->10.0.0.1:80-TCP',
        'source_ip': '172.16.0.99',
        'destination_ip': '10.0.0.1',
        'source_port': 65432,
        'destination_port': 80,
        'protocol': 'TCP',
        'duration': 0.05,  # 50ms duration (high rate flood)
        'bytes_sent': 250000,
        'bytes_recv': 0,
        'packets_sent': 500,
        'packets_recv': 0,
        'total_bytes': 250000,
        'total_packets': 500,
        'syn_count': 500,
        'ack_count': 0,
        'fin_count': 0,
        'rst_count': 0
    }

    # Process suspicious flow through manager
    manager._process_completed_flow(suspicious_flow)
    assert 'detection' in suspicious_flow, "Detection result missing from completed flow dictionary!"
    suspicious_result = suspicious_flow['detection']

    logger.info(f"Suspicious Flow Detection Output: {suspicious_result}")

    for field in required_fields:
        assert field in suspicious_result, f"Required normalized field '{field}' missing from output!"

    assert suspicious_result['prediction'] == 1, f"Suspicious traffic failed attack detection! Got {suspicious_result}"
    assert suspicious_result['attack_type'] != 'Normal', "Attack type for suspicious traffic should not be 'Normal'!"
    assert suspicious_result['confidence'] >= 0.5, f"Low confidence on attack detection: {suspicious_result['confidence']}"
    assert suspicious_result['severity'] in ['HIGH', 'CRITICAL', 'MEDIUM'], f"Low severity for attack flow: {suspicious_result['severity']}"

    # ---------------------------------------------------------
    # Test 3: Prediction Output Normalization Verification
    # ---------------------------------------------------------
    logger.info("\n[Step 3] Verifying Output Field Data Types & Constraints...")
    for res_name, res_dict in [("Benign", benign_result), ("Suspicious", suspicious_result)]:
        logger.info(f"\nVerifying {res_name} Result Normalized Format:")
        logger.info(f"  - prediction  : {res_dict['prediction']} (type: {type(res_dict['prediction']).__name__})")
        logger.info(f"  - attack_type : {res_dict['attack_type']} (type: {type(res_dict['attack_type']).__name__})")
        logger.info(f"  - confidence  : {res_dict['confidence']} (type: {type(res_dict['confidence']).__name__})")
        logger.info(f"  - severity    : {res_dict['severity']} (type: {type(res_dict['severity']).__name__})")
        logger.info(f"  - risk_score  : {res_dict['risk_score']} (type: {type(res_dict['risk_score']).__name__})")

        assert res_dict['prediction'] in [0, 1], "prediction must be 0 or 1"
        assert isinstance(res_dict['attack_type'], str), "attack_type must be string"
        assert 0.0 <= res_dict['confidence'] <= 1.0, "confidence must be between 0.0 and 1.0"
        assert isinstance(res_dict['severity'], str), "severity must be string"
        assert 0.0 <= res_dict['risk_score'] <= 100.0, "risk_score must be between 0.0 and 100.0"

    # ---------------------------------------------------------
    # Test 4: Alerts Logging Verification
    # ---------------------------------------------------------
    logger.info("\n[Step 4] Verifying Alerts Logged in LiveCaptureManager...")
    logger.info(f"Total alerts generated: {len(manager.alerts)}")
    assert len(manager.alerts) >= 1, "Expected at least 1 alert recorded for suspicious traffic!"
    alert = manager.alerts[0]
    logger.info(f"Logged Alert: {alert['attack_type']} | Severity: {alert['severity']} | Risk Score: {alert['risk_score']}")

    print("\n" + "=" * 60)
    print("ALL LIVE INFERENCE PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_pipeline_test()
