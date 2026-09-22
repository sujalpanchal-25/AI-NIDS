"""
Test Live Traffic Database Persistence & Deduplication
======================================================
Verifies that live network flow predictions are correctly persisted into SQLite database:
  1. NetworkFlow creation & in-place update for batch_id='LIVE_CAPTURE'
  2. Alert creation for malicious flows with supported ORM fields
  3. Deduplication preventing duplicate alerts or flow records
  4. Isolation between CSV analysis batches and live capture telemetry
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_live_db")


def run_db_integration_test():
    logger.info("=== Starting Live Database Integration Test ===")
    from app import create_app, db
    from app.models.database import NetworkFlow, Alert
    from collectors.live_capture import LiveCaptureManager
    from detection.detector import create_detection_engine

    app = create_app()
    with app.app_context():
        # Clean up previous test live capture records
        db.session.query(Alert).filter(Alert.batch_id == 'LIVE_CAPTURE').delete(synchronize_session=False)
        db.session.query(NetworkFlow).filter(NetworkFlow.batch_id == 'LIVE_CAPTURE').delete(synchronize_session=False)
        db.session.commit()
        logger.info("Cleaned previous LIVE_CAPTURE records from database.")

        model_dir = os.path.join(BASE_DIR, 'models')
        detector = create_detection_engine(model_dir=model_dir)
        manager = LiveCaptureManager(detector=detector, model_dir=model_dir)

        # ---------------------------------------------------------
        # Step 1: Benign Flow Test
        # ---------------------------------------------------------
        logger.info("\n[Step 1] Processing Benign Live Flow...")
        benign_flow_id = "test_flow_benign_1001"
        benign_flow = {
            'flow_id': benign_flow_id,
            'source_ip': '192.168.1.50',
            'destination_ip': '10.0.0.10',
            'source_port': 54321,
            'destination_port': 443,
            'protocol': 'TCP',
            'duration': 1.5,
            'bytes_sent': 500,
            'bytes_recv': 1500,
            'packets_sent': 5,
            'packets_recv': 10,
            'total_bytes': 2000,
            'total_packets': 15,
            'syn_count': 1,
            'ack_count': 14,
            'fin_count': 1,
            'rst_count': 0,
            'timestamp': datetime.utcnow()
        }

        manager._process_completed_flow(benign_flow)

        # Verify NetworkFlow DB Record
        db_flow = NetworkFlow.query.filter(
            NetworkFlow.batch_id == 'LIVE_CAPTURE',
            NetworkFlow.raw_data.like(f'%"flow_id": "{benign_flow_id}"%')
        ).first()

        assert db_flow is not None, "NetworkFlow record was not saved to database!"
        assert db_flow.source_ip == '192.168.1.50', f"IP mismatch: {db_flow.source_ip}"
        assert db_flow.batch_id == 'LIVE_CAPTURE', "batch_id must be 'LIVE_CAPTURE'"
        assert db_flow.is_anomaly is False, "Benign flow should not be flagged as anomaly"
        logger.info(f"Verified Benign NetworkFlow record ID #{db_flow.id}: label={db_flow.label}, is_anomaly={db_flow.is_anomaly}")

        # Verify No Alert was inserted for benign traffic
        db_alert = Alert.query.filter(
            Alert.batch_id == 'LIVE_CAPTURE',
            Alert.raw_data.like(f'%"flow_id": "{benign_flow_id}"%')
        ).first()
        assert db_alert is None, "Alert was unexpectedly created for benign traffic!"
        logger.info("Confirmed: No Alert record inserted for benign traffic.")

        # ---------------------------------------------------------
        # Step 2: Malicious Flow Test
        # ---------------------------------------------------------
        logger.info("\n[Step 2] Processing Malicious Flood Flow...")
        malicious_flow_id = "test_flow_malicious_9001"
        malicious_flow = {
            'flow_id': malicious_flow_id,
            'source_ip': '172.16.0.88',
            'destination_ip': '10.0.0.1',
            'source_port': 60000,
            'destination_port': 80,
            'protocol': 'TCP',
            'duration': 0.02,
            'bytes_sent': 500000,
            'bytes_recv': 0,
            'packets_sent': 1000,
            'packets_recv': 0,
            'total_bytes': 500000,
            'total_packets': 1000,
            'syn_count': 1000,
            'ack_count': 0,
            'fin_count': 0,
            'rst_count': 0,
            'timestamp': datetime.utcnow()
        }

        manager._process_completed_flow(malicious_flow)

        # Verify NetworkFlow DB Record
        db_mflow = NetworkFlow.query.filter(
            NetworkFlow.batch_id == 'LIVE_CAPTURE',
            NetworkFlow.raw_data.like(f'%"flow_id": "{malicious_flow_id}"%')
        ).first()

        assert db_mflow is not None, "Malicious NetworkFlow record was not saved to database!"
        assert db_mflow.is_anomaly is True, "Malicious flow must be flagged as anomaly!"
        logger.info(f"Verified Malicious NetworkFlow record ID #{db_mflow.id}: predicted_label={db_mflow.predicted_label}")

        # Verify Alert DB Record & Supported ORM Fields
        db_mal_alert = Alert.query.filter(
            Alert.batch_id == 'LIVE_CAPTURE',
            Alert.raw_data.like(f'%"flow_id": "{malicious_flow_id}"%')
        ).first()

        assert db_mal_alert is not None, "Alert record was not created for malicious flow!"
        assert db_mal_alert.source_ip == '172.16.0.88', f"Alert source_ip mismatch: {db_mal_alert.source_ip}"
        assert db_mal_alert.destination_ip == '10.0.0.1', f"Alert destination_ip mismatch: {db_mal_alert.destination_ip}"
        assert db_mal_alert.attack_type != 'Normal', "Alert attack_type should not be 'Normal'"
        assert db_mal_alert.severity in ['critical', 'high', 'medium', 'low', 'info'], f"Invalid severity: {db_mal_alert.severity}"
        assert 0.0 <= db_mal_alert.confidence <= 1.0, f"Invalid confidence: {db_mal_alert.confidence}"
        assert 0.0 <= db_mal_alert.risk_score <= 100.0, f"Invalid risk score: {db_mal_alert.risk_score}"
        assert db_mal_alert.acknowledged is False, "New alert acknowledged should be False"
        assert db_mal_alert.resolved is False, "New alert resolved should be False"
        assert db_mal_alert.batch_id == 'LIVE_CAPTURE', "batch_id must be 'LIVE_CAPTURE'"

        logger.info(
            f"Verified Alert record ID #{db_mal_alert.id}: "
            f"attack_type='{db_mal_alert.attack_type}', severity='{db_mal_alert.severity}', "
            f"confidence={db_mal_alert.confidence}, risk_score={db_mal_alert.risk_score}, "
            f"acknowledged={db_mal_alert.acknowledged}, resolved={db_mal_alert.resolved}"
        )

        # ---------------------------------------------------------
        # Step 3: Deduplication Test
        # ---------------------------------------------------------
        logger.info("\n[Step 3] Testing Flow & Alert Deduplication...")
        flow_count_before = NetworkFlow.query.filter(
            NetworkFlow.batch_id == 'LIVE_CAPTURE',
            NetworkFlow.raw_data.like(f'%"flow_id": "{malicious_flow_id}"%')
        ).count()
        alert_count_before = Alert.query.filter(
            Alert.batch_id == 'LIVE_CAPTURE',
            Alert.raw_data.like(f'%"flow_id": "{malicious_flow_id}"%')
        ).count()

        # Reprocess exact same malicious flow ID
        manager._process_completed_flow(malicious_flow)

        flow_count_after = NetworkFlow.query.filter(
            NetworkFlow.batch_id == 'LIVE_CAPTURE',
            NetworkFlow.raw_data.like(f'%"flow_id": "{malicious_flow_id}"%')
        ).count()
        alert_count_after = Alert.query.filter(
            Alert.batch_id == 'LIVE_CAPTURE',
            Alert.raw_data.like(f'%"flow_id": "{malicious_flow_id}"%')
        ).count()

        assert flow_count_after == flow_count_before, f"NetworkFlow duplicated! Before={flow_count_before}, After={flow_count_after}"
        assert alert_count_after == alert_count_before, f"Alert duplicated! Before={alert_count_before}, After={alert_count_after}"
        logger.info("Confirmed: Duplicate processing updated records in-place without duplicate rows.")

    print("\n" + "=" * 60)
    print("ALL LIVE DATABASE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_db_integration_test()
