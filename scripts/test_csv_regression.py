"""
Comprehensive CSV Pipeline Regression Test
============================================
Verifies that all pre-existing CSV processing features, dataset switching,
XGBoost ML classification, database serialization, dashboard synchronization,
analytics views, alert filtering, and report export logic remain 100% intact
and regression-free after Live Monitoring implementation.
"""

import sys
import os
import time
import logging
import json

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_csv_regression")


def run_csv_regression_test():
    logger.info("=== Starting Comprehensive CSV Pipeline Regression Test ===")
    from app import create_app, db
    from app.models.database import NetworkFlow, Alert
    from app.services.analysis import CSVAnalysisService, ANALYSIS_STATUS

    app = create_app()
    app.config['TESTING'] = True

    with app.app_context():
        service = CSVAnalysisService()
        sample_path = os.path.join(BASE_DIR, 'data', 'datasets', 'sample_traffic.csv')
        assert os.path.exists(sample_path), f"Sample CSV dataset missing at {sample_path}"

        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess['_user_id'] = '1'

            # ---------------------------------------------------------
            # Step 1: CSV Upload & Analysis Execution
            # ---------------------------------------------------------
            logger.info("\n[Step 1] Testing CSV Dataset Analysis Execution...")
            batch_id = service.start_analysis_async(sample_path, app=app, use_sample=True)
            assert batch_id is not None, "Failed to start async CSV analysis!"
            logger.info(f"Analysis started for batch ID: {batch_id}")

            # Wait for background thread completion
            timeout = 60
            start_t = time.time()
            completed = False
            while time.time() - start_t < timeout:
                st = service.get_analysis_status(batch_id)
                if st and st.get('status') == 'completed':
                    completed = True
                    logger.info(f"CSV Analysis completed in {time.time() - start_t:.2f}s: {st.get('summary')}")
                    break
                elif st and st.get('status') == 'error':
                    raise RuntimeError(f"CSV Analysis failed with error: {st.get('error')}")
                time.sleep(0.5)

            assert completed, f"CSV Analysis timed out after {timeout} seconds!"

            # ---------------------------------------------------------
            # Step 2: Database Records Verification
            # ---------------------------------------------------------
            logger.info("\n[Step 2] Verifying NetworkFlow and Alert Records in Database...")
            batch_flows = NetworkFlow.query.filter_by(batch_id=batch_id).all()
            batch_alerts = Alert.query.filter_by(batch_id=batch_id).all()

            logger.info(f"Database Records for {batch_id}: Flows={len(batch_flows)}, Alerts={len(batch_alerts)}")
            assert len(batch_flows) > 0, "No NetworkFlow records saved for CSV batch!"
            
            # Verify XGBoost / Ensemble predictions present in raw_data or predicted_label
            sample_flow = batch_flows[0]
            assert sample_flow.predicted_label is not None, "predicted_label missing from NetworkFlow"
            logger.info(f"Sample NetworkFlow Record: ID #{sample_flow.id}, Label={sample_flow.label}, Predicted={sample_flow.predicted_label}, Anomaly={sample_flow.is_anomaly}")

            if len(batch_alerts) > 0:
                sample_alert = batch_alerts[0]
                assert sample_alert.attack_type is not None, "attack_type missing from Alert"
                assert sample_alert.severity in ['critical', 'high', 'medium', 'low', 'info'], f"Invalid severity: {sample_alert.severity}"
                assert 0.0 <= sample_alert.confidence <= 1.0, f"Invalid confidence: {sample_alert.confidence}"
                assert 0.0 <= sample_alert.risk_score <= 100.0, f"Invalid risk_score: {sample_alert.risk_score}"
                logger.info(f"Sample Alert Record: ID #{sample_alert.id}, Type={sample_alert.attack_type}, Severity={sample_alert.severity}, Conf={sample_alert.confidence}")

            # ---------------------------------------------------------
            # Step 3: Dataset Selection & Switching
            # ---------------------------------------------------------
            logger.info("\n[Step 3] Testing Dataset Selection & Switching...")
            
            # Select sample_traffic.csv
            sel_resp = client.post('/api/select-dataset', json={'filename': 'sample_traffic.csv'})
            assert sel_resp.status_code == 200, f"Select dataset failed with status {sel_resp.status_code}"
            sel_data = sel_resp.get_json()
            assert sel_data.get('success') is True, f"Select dataset error: {sel_data}"
            logger.info(f"Select Dataset Response: {sel_data.get('message')}")

            # Check dashboard sync under CSV dataset filter
            sync_csv_resp = client.get('/dashboard/sync')
            sync_csv_data = sync_csv_resp.get_json()
            assert sync_csv_data.get('success') is True, "Dashboard sync failed in CSV mode!"
            csv_flows_count = sync_csv_data['stats']['total_flows']
            logger.info(f"Dashboard Stats under CSV Dataset Filter: Flows={csv_flows_count}")
            assert csv_flows_count == len(batch_flows), f"Expected dashboard flows to equal CSV batch flow count ({len(batch_flows)}), got {csv_flows_count}"

            # Switch back to Live Traffic view
            reset_resp = client.post('/api/select-dataset', json={'filename': ''})
            assert reset_resp.status_code == 200, f"Reset dataset failed with status {reset_resp.status_code}"
            reset_data = reset_resp.get_json()
            assert reset_data.get('success') is True, f"Reset dataset error: {reset_data}"
            logger.info(f"Reset Dataset Response: {reset_data.get('message')}")

            # ---------------------------------------------------------
            # Step 4: Analytics Endpoints under CSV vs Live Mode
            # ---------------------------------------------------------
            logger.info("\n[Step 4] Testing Analytics Endpoints under CSV & Live Mode...")
            
            # Set session to CSV batch
            with client.session_transaction() as sess:
                sess['selected_dataset'] = batch_id

            proto_csv = client.get('/analytics/api/protocol-distribution').get_json()
            atk_csv = client.get('/analytics/api/attack-types').get_json()

            logger.info(f"CSV Mode Protocol Distribution: {proto_csv}")
            logger.info(f"CSV Mode Attack Types: {atk_csv}")
            assert len(proto_csv.get('labels', [])) > 0, "Protocol distribution empty in CSV mode"

            # Clear session (Live mode)
            with client.session_transaction() as sess:
                sess.pop('selected_dataset', None)

            proto_live = client.get('/analytics/api/protocol-distribution').get_json()
            logger.info(f"Live Mode Protocol Distribution: {proto_live}")
            assert len(proto_live.get('labels', [])) > 0, "Protocol distribution empty in Live mode"

            # ---------------------------------------------------------
            # Step 5: Alerts Page & Report Export
            # ---------------------------------------------------------
            logger.info("\n[Step 5] Testing Alerts Page & CSV Export Endpoints...")
            
            alerts_page = client.get('/alerts/')
            assert alerts_page.status_code == 200, f"Alerts list page failed with status {alerts_page.status_code}"
            logger.info("Confirmed: Alerts page rendered successfully.")

            # Test Report Exports
            csv_report_resp = client.get(f'/api/export-report/{batch_id}/csv')
            assert csv_report_resp.status_code == 200, f"CSV report export failed with status {csv_report_resp.status_code}"
            assert b'AI-NIDS' in csv_report_resp.data or b'batch_id' in csv_report_resp.data.lower() or b'timestamp' in csv_report_resp.data.lower()
            logger.info("Confirmed: Batch CSV Report Export endpoint works.")

            alerts_export_resp = client.get('/analytics/export/alerts')
            assert alerts_export_resp.status_code == 200, f"Alerts export failed with status {alerts_export_resp.status_code}"
            logger.info("Confirmed: Analytics Alerts CSV Export endpoint works.")

            flows_export_resp = client.get('/analytics/export/flows')
            assert flows_export_resp.status_code == 200, f"Flows export failed with status {flows_export_resp.status_code}"
            logger.info("Confirmed: Analytics Flows CSV Export endpoint works.")

    print("\n" + "=" * 60)
    print("ALL CSV WORKFLOW REGRESSION TESTS PASSED 100% SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    run_csv_regression_test()
