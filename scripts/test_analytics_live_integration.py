"""
Test Analytics Pages & API Endpoints Live Traffic Integration
================================================================
Verifies that:
  1. Live NetworkFlow records (batch_id='LIVE_CAPTURE') appear in Traffic Analytics & Protocol Distribution
  2. Live Alert records (batch_id='LIVE_CAPTURE') appear in Threat Analytics, Severity Distribution, Attack Types, & Timelines
  3. CSV dataset selection mode (session['selected_dataset']) strictly filters by batch ID
"""

import sys
import os
import logging
import json
from datetime import datetime

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_analytics_live")


def run_analytics_live_test():
    logger.info("=== Starting Analytics Live Traffic Verification Test ===")
    from app import create_app, db
    from app.models.database import NetworkFlow, Alert
    from collectors.live_capture import LiveCaptureManager
    from detection.detector import create_detection_engine

    app = create_app()
    app.config['TESTING'] = True

    with app.app_context():
        # Insert test Live NetworkFlow and Alert records
        detector = create_detection_engine(model_dir=os.path.join(BASE_DIR, 'models'))
        manager = LiveCaptureManager(detector=detector, model_dir=os.path.join(BASE_DIR, 'models'))

        benign_flow = {
            'flow_id': 'analytics_live_test_benign_5001',
            'source_ip': '192.168.1.199',
            'destination_ip': '10.0.0.50',
            'source_port': 51111,
            'destination_port': 443,
            'protocol': 'UDP',
            'duration': 1.0,
            'bytes_sent': 500,
            'bytes_recv': 1500,
            'packets_sent': 5,
            'packets_recv': 10,
            'total_bytes': 2000,
            'total_packets': 15,
            'syn_count': 0,
            'ack_count': 0,
            'fin_count': 0,
            'rst_count': 0,
            'timestamp': datetime.utcnow()
        }
        manager._process_completed_flow(benign_flow)

        malicious_flow = {
            'flow_id': 'analytics_live_test_malicious_6001',
            'source_ip': '172.16.88.99',
            'destination_ip': '10.0.0.1',
            'source_port': 61111,
            'destination_port': 80,
            'protocol': 'TCP',
            'duration': 0.01,
            'bytes_sent': 600000,
            'bytes_recv': 0,
            'packets_sent': 1200,
            'packets_recv': 0,
            'total_bytes': 600000,
            'total_packets': 1200,
            'syn_count': 1200,
            'ack_count': 0,
            'fin_count': 0,
            'rst_count': 0,
            'timestamp': datetime.utcnow()
        }
        manager._process_completed_flow(malicious_flow)

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess['_user_id'] = '1'
                sess.pop('selected_dataset', None)  # Ensure Live Mode

            # ---------------------------------------------------------
            # Test 1: Live Mode Analytics Visibility
            # ---------------------------------------------------------
            logger.info("\n[Step 1] Verifying Live Mode Analytics Endpoints...")

            # Protocol Distribution
            proto_resp = client.get('/analytics/api/protocol-distribution')
            assert proto_resp.status_code == 200, f"Protocol endpoint status {proto_resp.status_code}"
            proto_data = proto_resp.get_json()
            logger.info(f"Protocol Distribution in Live Mode: {proto_data}")
            assert 'UDP' in proto_data.get('labels', []) or 'TCP' in proto_data.get('labels', []), "Protocols missing from Live analytics"

            # Attack Types
            atk_resp = client.get('/analytics/api/attack-types')
            assert atk_resp.status_code == 200, f"Attack types status {atk_resp.status_code}"
            atk_data = atk_resp.get_json()
            logger.info(f"Attack Types in Live Mode: {atk_data}")
            assert len(atk_data.get('labels', [])) > 0, "Attack types should not be empty when live alert exists"

            # Severity Distribution
            sev_resp = client.get('/analytics/api/severity-distribution')
            assert sev_resp.status_code == 200, f"Severity status {sev_resp.status_code}"
            sev_data = sev_resp.get_json()
            logger.info(f"Severity Distribution in Live Mode: {sev_data}")
            assert sum(sev_data.get('values', [])) > 0, "Severity count should be > 0"

            # Timeline
            timeline_resp = client.get('/analytics/api/timeline?metric=alerts')
            assert timeline_resp.status_code == 200, f"Timeline status {timeline_resp.status_code}"
            timeline_data = timeline_resp.get_json()
            logger.info(f"Timeline Alerts in Live Mode: {timeline_data}")
            assert sum(timeline_data.get('values', [])) > 0, "Alert timeline should contain alerts"

            # Threat Analytics HTML Page
            threat_page_resp = client.get('/analytics/threats')
            assert threat_page_resp.status_code == 200, f"Threat analytics status {threat_page_resp.status_code}"
            assert b'Threat Analysis' in threat_page_resp.data or b'threat' in threat_page_resp.data.lower()
            logger.info("Confirmed: Threat Analytics HTML page rendered successfully in Live Mode.")

            # Traffic Analytics HTML Page
            traffic_page_resp = client.get('/analytics/traffic')
            assert traffic_page_resp.status_code == 200, f"Traffic analytics status {traffic_page_resp.status_code}"
            assert b'Traffic Analysis' in traffic_page_resp.data or b'traffic' in traffic_page_resp.data.lower()
            logger.info("Confirmed: Traffic Analytics HTML page rendered successfully in Live Mode.")

            # ---------------------------------------------------------
            # Test 2: CSV Selected Dataset Mode Isolation
            # ---------------------------------------------------------
            logger.info("\n[Step 2] Verifying CSV Dataset Selected Mode Isolation...")
            with client.session_transaction() as sess:
                sess['selected_dataset'] = 'batch-non-existent-uuid-9999'

            # Protocol Distribution with non-matching batch_id should return 0/empty
            csv_proto_resp = client.get('/analytics/api/protocol-distribution')
            csv_proto_data = csv_proto_resp.get_json()
            logger.info(f"Protocol Distribution in CSV Filter Mode: {csv_proto_data}")
            assert len(csv_proto_data.get('labels', [])) == 0, "CSV mode should isolate and exclude live records!"

            # Attack Types in CSV mode with non-matching batch_id
            csv_atk_resp = client.get('/analytics/api/attack-types')
            csv_atk_data = csv_atk_resp.get_json()
            logger.info(f"Attack Types in CSV Filter Mode: {csv_atk_data}")
            assert len(csv_atk_data.get('labels', [])) == 0, "CSV mode should exclude live threat records!"

            logger.info("Confirmed: CSV Selected Mode strictly filters by batch ID and isolates live records.")

    print("\n" + "=" * 60)
    print("ALL ANALYTICS LIVE TRAFFIC VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_analytics_live_test()
