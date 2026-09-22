"""
Test Dashboard Live Monitoring Integration & API Endpoints
=============================================================
Verifies that dashboard live monitoring API endpoints and backend metrics function seamlessly:
  1. GET /api/live/interfaces returns valid network adapter choices
  2. POST /api/live/start initiates capture state and returns success
  3. GET /api/live/status returns real backend metrics (packets_captured, flows_processed, threats_detected, benign_count, last_threat, last_update)
  4. POST /api/live/stop stops capture cleanly
  5. POST /api/select-dataset with filename="" resets session to live traffic mode
"""

import sys
import os
import logging
import json

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_dashboard_live")


def run_dashboard_live_test():
    logger.info("=== Starting Dashboard Live Monitoring API Test ===")
    from app import create_app

    app = create_app()
    app.config['TESTING'] = True

    with app.test_client() as client:
        # Simulate authenticated session
        with client.session_transaction() as sess:
            sess['_user_id'] = '1'

        # ---------------------------------------------------------
        # Test 1: Fetch Network Interfaces
        # ---------------------------------------------------------
        logger.info("\n[Step 1] Testing GET /api/live/interfaces...")
        resp = client.get('/api/live/interfaces')
        assert resp.status_code == 200, f"Interfaces endpoint failed with status {resp.status_code}"
        data = resp.get_json()
        assert data.get('success') is True, f"Response success is False: {data}"
        assert 'interfaces' in data, "interfaces key missing from response"
        logger.info(f"Available interfaces found: {len(data['interfaces'])} (Default: {data.get('default_interface')})")

        # ---------------------------------------------------------
        # Test 2: Start Monitoring
        # ---------------------------------------------------------
        logger.info("\n[Step 2] Testing POST /api/live/start...")
        start_resp = client.post('/api/live/start', json={'interface': data.get('default_interface', ''), 'filter': 'ip'})
        assert start_resp.status_code == 200, f"Start monitoring failed with status {start_resp.status_code}"
        start_data = start_resp.get_json()
        assert start_data.get('success') is True, f"Start failed: {start_data}"
        logger.info(f"Start Response: {start_data.get('message')}")

        # ---------------------------------------------------------
        # Test 3: Status & Live Metrics
        # ---------------------------------------------------------
        logger.info("\n[Step 3] Testing GET /api/live/status...")
        status_resp = client.get('/api/live/status')
        assert status_resp.status_code == 200, f"Status endpoint failed with status {status_resp.status_code}"
        status_data = status_resp.get_json()
        assert status_data.get('success') is True, f"Status failed: {status_data}"
        assert status_data.get('is_running') is True, "Capture should be running after start!"
        assert status_data.get('status') == 'MONITORING', f"Expected status MONITORING, got {status_data.get('status')}"

        stats = status_data.get('statistics', {})
        logger.info(f"Live Status Stats: {json.dumps(stats, indent=2)}")

        required_metric_keys = ['packets_captured', 'flows_processed', 'threats_detected', 'benign_count', 'last_threat', 'last_update']
        for key in required_metric_keys:
            assert key in stats, f"Required live metric '{key}' missing from status response!"

        # ---------------------------------------------------------
        # Test 4: Stop Monitoring
        # ---------------------------------------------------------
        logger.info("\n[Step 4] Testing POST /api/live/stop...")
        stop_resp = client.post('/api/live/stop')
        assert stop_resp.status_code == 200, f"Stop monitoring failed with status {stop_resp.status_code}"
        stop_data = stop_resp.get_json()
        assert stop_data.get('success') is True, f"Stop failed: {stop_data}"

        # Verify status changed to STOPPED
        status_after_stop = client.get('/api/live/status').get_json()
        assert status_after_stop.get('is_running') is False, "Capture should not be running after stop!"
        assert status_after_stop.get('status') == 'STOPPED', f"Expected status STOPPED, got {status_after_stop.get('status')}"
        logger.info("Confirmed: Capture stopped cleanly.")

        # ---------------------------------------------------------
        # Test 5: Dataset Reset (Live Traffic Mode)
        # ---------------------------------------------------------
        logger.info("\n[Step 5] Testing POST /api/select-dataset with filename=''...")
        select_resp = client.post('/api/select-dataset', json={'filename': ''})
        assert select_resp.status_code == 200, f"Select dataset reset failed with status {select_resp.status_code}"
        select_data = select_resp.get_json()
        assert select_data.get('success') is True, f"Select dataset failed: {select_data}"
        assert 'Reset to live traffic view' in select_data.get('message', ''), "Expected reset message"
        logger.info("Confirmed: Dataset selection reset cleanly to live traffic view.")

    print("\n" + "=" * 60)
    print("ALL DASHBOARD LIVE MONITORING TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_dashboard_live_test()
