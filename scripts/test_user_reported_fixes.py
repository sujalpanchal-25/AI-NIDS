import os
import sys

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_user_reported_fixes():
    print("=" * 70)
    print("AI-NIDS VERIFICATION: USER REPORTED ISSUES (GNN DEVICE, 404s)")
    print("=" * 70)

    # 1. GNN device property & inference
    print("\n[1/4] Testing GNNIntrusionDetector device & predict_flow_anomaly...")
    from ml.models.gnn_detector import create_gnn_detector
    gnn_path = os.path.join(project_root, 'models', 'gnn_model.pt')
    gnn = create_gnn_detector(pretrained_path=gnn_path if os.path.exists(gnn_path) else None)
    
    print(f" -> GNN device property: {gnn.device}")
    assert hasattr(gnn, 'device'), "GNNIntrusionDetector missing device property!"

    sample_flow = {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.5',
        'src_port': 4433,
        'dst_port': 80,
        'protocol': 'TCP',
        'bytes_sent': 1200,
        'bytes_recv': 3400,
        'packets_sent': 8,
        'packets_recv': 10,
        'duration': 0.5
    }
    score = gnn.predict_flow_anomaly(sample_flow)
    print(f" -> GNN flow anomaly score: {score:.4f}")
    assert score is not None and score >= 0.0, "GNN prediction failed!"

    # 2. Flask app routes & static assets
    print("\n[2/4] Testing Flask endpoints and static assets...")
    from app import create_app
    app = create_app('testing')
    client = app.test_client()

    # Test GET /api/v1/sources/top?limit=100
    print("\n[3/4] Testing GET /api/v1/sources/top?limit=100...")
    res_sources = client.get('/api/v1/sources/top?limit=100')
    print(f" -> Status: {res_sources.status_code}")
    print(f" -> Data: {res_sources.get_json()}")
    assert res_sources.status_code == 200, f"Expected 200, got {res_sources.status_code}"

    # Test GET /static/js/charts.js & GET /static/images/icons/icon-144x144.png
    print("\n[4/4] Testing static assets (charts.js, PWA icons)...")
    res_charts = client.get('/static/js/charts.js')
    print(f" -> GET /static/js/charts.js Status: {res_charts.status_code}")
    assert res_charts.status_code == 200, f"charts.js returned {res_charts.status_code}"

    res_icon = client.get('/static/images/icons/icon-144x144.png')
    print(f" -> GET /static/images/icons/icon-144x144.png Status: {res_icon.status_code}")
    assert res_icon.status_code == 200, f"PWA icon returned {res_icon.status_code}"

    print("\n" + "=" * 70)
    print("✅ SUCCESS: ALL REPORTED ISSUES FIXED AND VERIFIED!")
    print("=" * 70)

if __name__ == '__main__':
    test_user_reported_fixes()
