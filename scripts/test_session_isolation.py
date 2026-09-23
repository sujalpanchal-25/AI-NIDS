import os
import sys
import unittest
from datetime import datetime

# Set path
sys.path.insert(0, os.path.abspath('.'))

from app import create_app, db
from app.models.database import User, NetworkFlow, Alert

class TestDashboardZeroStart(unittest.TestCase):
    def setUp(self):
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.create_all()

        # Create test user
        self.user = User.query.filter_by(username='test_analyst').first()
        if not self.user:
            self.user = User(
                username='test_analyst',
                email='test@example.com',
                role='analyst',
                is_active=True,
                is_verified=True
            )
            self.user.set_password('Password123!')
            db.session.add(self.user)
            db.session.commit()

    def tearDown(self):
        self.ctx.pop()

    def test_login_and_dashboard_zero_metrics(self):
        # 1. Login
        login_resp = self.client.post('/auth/login', data={
            'username': 'test_analyst',
            'password': 'Password123!'
        }, follow_redirects=True)
        self.assertEqual(login_resp.status_code, 200)

        # 2. Query dashboard stats
        stats_resp = self.client.get('/dashboard/stats')
        self.assertEqual(stats_resp.status_code, 200)
        stats = stats_resp.get_json()

        print("\n--- Fresh Login Stats ---")
        print(stats)
        self.assertEqual(stats['total_flows'], 0)
        self.assertEqual(stats['total_alerts'], 0)

        # 3. Simulate switching to Live Traffic
        select_resp = self.client.post('/api/select-dataset', json={'filename': ''})
        self.assertEqual(select_resp.status_code, 200)

        stats_after_live = self.client.get('/dashboard/stats').get_json()
        print("\n--- Live Traffic Reset Stats ---")
        print(stats_after_live)
        self.assertEqual(stats_after_live['total_flows'], 0)
        self.assertEqual(stats_after_live['total_alerts'], 0)

    def test_clear_data_button_endpoint(self):
        # 1. Login
        self.client.post('/auth/login', data={
            'username': 'test_analyst',
            'password': 'Password123!'
        }, follow_redirects=True)

        # 2. Insert dummy flow & alert
        flow = NetworkFlow(
            source_ip='192.168.1.50',
            destination_ip='10.0.0.1',
            source_port=54321,
            destination_port=80,
            protocol='TCP',
            total_bytes=1024,
            is_anomaly=True
        )
        alert = Alert(
            source_ip='192.168.1.50',
            destination_ip='10.0.0.1',
            attack_type='DDoS',
            severity='critical',
            confidence=0.98
        )
        db.session.add(flow)
        db.session.add(alert)
        db.session.commit()

        # 3. Call Clear Data Endpoint
        clear_resp = self.client.post('/api/clear-data')
        self.assertEqual(clear_resp.status_code, 200)
        clear_data = clear_resp.get_json()
        self.assertTrue(clear_data.get('success'))

        # 4. Check that DB records and dashboard stats are 0
        self.assertEqual(NetworkFlow.query.count(), 0)
        self.assertEqual(Alert.query.count(), 0)

        stats = self.client.get('/dashboard/stats').get_json()
        self.assertEqual(stats['total_flows'], 0)
        self.assertEqual(stats['total_alerts'], 0)
        print("\n--- Stats after executing Clear Data ---")
        print(stats)

    def test_prediction_history_endpoint(self):
        # 1. Login
        self.client.post('/auth/login', data={
            'username': 'test_analyst',
            'password': 'Password123!'
        }, follow_redirects=True)

        # 2. Query /api/prediction-history
        hist_resp = self.client.get('/api/prediction-history')
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.get_json()

        self.assertTrue(hist_data.get('success'))
        self.assertIn('datasets', hist_data)
        self.assertIn('live_stats', hist_data)
        self.assertIn('active_mode', hist_data)
        print("\n--- Prediction History API Response ---")
        print("Active mode:", hist_data.get('active_mode'))
        print("Datasets found:", len(hist_data.get('datasets', [])))
        print("Live stats:", hist_data.get('live_stats'))

if __name__ == '__main__':
    unittest.main()
