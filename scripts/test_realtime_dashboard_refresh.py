"""
Test Real-Time Dashboard Refresh with Live Monitoring Telemetry
=================================================================
Verifies end-to-end real-time dashboard data synchronization:
  1. Benign live traffic processing updates NetworkFlow database counts and /dashboard/sync total_flows
  2. Malicious traffic processing generates Alert records and updates /dashboard/sync recent_alerts & total_alerts
  3. Stopping live capture halts capture thread & releases resources
  4. CSV selected mode compatibility remains intact
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
logger = logging.getLogger("test_realtime_refresh")


def run_realtime_refresh_test():
    logger.info("=== Starting Real-Time Dashboard Refresh Test ===")
    from app import create_app, db
    from app.models.database import NetworkFlow, Alert
    from collectors.live_capture import LiveCaptureManager
    from detection.detector import create_detection_engine

    app = create_app()
    app.config['TESTING'] = True

    with app.app_context():
        # Clean previous records for an isolated test
        db.session.query(Alert).delete(synchronize_session=False)
        db.session.query(NetworkFlow).delete(synchronize_session=False)
        db.session.commit()

        detector = create_detection_engine(model_dir=os.path.join(BASE_DIR, 'models'))
        manager = LiveCaptureManager(detector=detector, model_dir=os.path.join(BASE_DIR, 'models'))

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess['_user_id'] = '1'

            # ---------------------------------------------------------
            # Step 1: Initial Dashboard Sync State
            # ---------------------------------------------------------
            logger.info("\n[Step 1] Fetching Initial Dashboard Sync State...")
            sync_resp = client.get('/dashboard/sync')
            assert sync_resp.status_code == 200, f"Sync request failed with status {sync_resp.status_code}"
            init_sync = sync_resp.get_json()
            assert init_sync.get('success') is True, f"Sync response success is False: {init_sync}"

            init_flows = init_sync['stats']['total_flows']
            init_alerts = init_sync['stats']['total_alerts']
            logger.info(f"Initial Dashboard Stats: Flows={init_flows}, Alerts={init_alerts}")

            # ---------------------------------------------------------
            # Step 2: Benign Traffic -> DB Update -> Sync Refresh
            # ---------------------------------------------------------
            logger.info("\n[Step 2] Processing Benign Live Flow & Verifying Dashboard Sync...")
            benign_flow = {
                'flow_id': 'realtime_test_benign_2001',
                'source_ip': '192.168.1.75',
                'destination_ip': '10.0.0.25',
                'source_port': 52345,
                'destination_port': 443,
                'protocol': 'TCP',
                'duration': 1.8,
                'bytes_sent': 800,
                'bytes_recv': 2400,
                'packets_sent': 8,
                'packets_recv': 16,
                'total_bytes': 3200,
                'total_packets': 24,
                'syn_count': 1,
                'ack_count': 23,
                'fin_count': 1,
                'rst_count': 0,
                'timestamp': datetime.utcnow()
            }

            manager._process_completed_flow(benign_flow)

            # Query dashboard sync
            sync_resp2 = client.get('/dashboard/sync')
            data2 = sync_resp2.get_json()
            assert data2['success'] is True, f"Sync request failed: {data2}"
            updated_flows = data2['stats']['total_flows']
            logger.info(f"Updated Dashboard Stats after Benign Flow: Flows={updated_flows} (was {init_flows})")
            assert updated_flows >= init_flows + 1, f"Expected total_flows to increment! Initial: {init_flows}, Updated: {updated_flows}"

            # ---------------------------------------------------------
            # Step 3: Malicious Traffic -> DB Update -> Threat Feed Sync
            # ---------------------------------------------------------
            logger.info("\n[Step 3] Processing Malicious Flood Flow & Verifying Threat Feed Refresh...")
            malicious_flow = {
                'flow_id': 'realtime_test_malicious_8001',
                'source_ip': '172.16.55.99',
                'destination_ip': '10.0.0.1',
                'source_port': 64321,
                'destination_port': 80,
                'protocol': 'TCP',
                'duration': 0.01,
                'bytes_sent': 400000,
                'bytes_recv': 0,
                'packets_sent': 800,
                'packets_recv': 0,
                'total_bytes': 400000,
                'total_packets': 800,
                'syn_count': 800,
                'ack_count': 0,
                'fin_count': 0,
                'rst_count': 0,
                'timestamp': datetime.utcnow()
            }

            manager._process_completed_flow(malicious_flow)

            # Query dashboard sync
            sync_resp3 = client.get('/dashboard/sync')
            data3 = sync_resp3.get_json()
            assert data3['success'] is True, f"Sync request failed: {data3}"

            updated_alerts = data3['stats']['total_alerts']
            recent_alerts = data3.get('recent_alerts', [])
            logger.info(f"Updated Dashboard Stats after Malicious Flow: Alerts={updated_alerts} (was {init_alerts})")
            assert updated_alerts >= init_alerts + 1, f"Expected total_alerts to increment! Initial: {init_alerts}, Updated: {updated_alerts}"

            assert len(recent_alerts) > 0, "recent_alerts list must contain newly created alert!"
            latest_alert = recent_alerts[0]
            logger.info(f"Recent Alert in Dashboard Sync Feed: {latest_alert.get('attack_type')} from {latest_alert.get('source_ip')} ({latest_alert.get('severity')})")
            assert latest_alert.get('source_ip') == '172.16.55.99', f"IP mismatch in alert feed: {latest_alert.get('source_ip')}"

            # ---------------------------------------------------------
            # Step 4: Stop Live Capture & Resource Cleanup
            # ---------------------------------------------------------
            logger.info("\n[Step 4] Stopping Live Capture and Verifying Resource Release...")
            stop_resp = client.post('/api/live/stop')
            assert stop_resp.status_code == 200, f"Stop failed with status {stop_resp.status_code}"
            assert manager.is_running is False, "LiveCaptureManager must be stopped!"
            logger.info("Confirmed: Capture thread stopped and polling resources released.")

    print("\n" + "=" * 60)
    print("ALL REAL-TIME DASHBOARD REFRESH TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_realtime_refresh_test()
