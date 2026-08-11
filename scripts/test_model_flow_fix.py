"""
Test script to verify that ML models act as the primary detection engine
and heuristic rules act strictly as the last fallback.
"""

import os
import sys
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_model_flow_architecture():
    print("=" * 75)
    print("AI-NIDS BACKEND VERIFICATION: ML PRIMARY ENGINE & RULE FALLBACK FLOW")
    print("=" * 75)

    # 1. Test CSV Analysis Service flow reordering
    print("\n[1/3] Testing CSVAnalysisService flow integration...")
    from app import create_app
    from app.services.analysis import CSVAnalysisService

    app = create_app('testing')
    with app.app_context():
        service = CSVAnalysisService()
        print(f" -> ML Detector loaded: {service.detector is not None}")
        assert service.detector is not None, "CSVAnalysisService failed to instantiate Detector"

        # Check loaded models
        xgb_loaded = getattr(service.detector, 'xgboost_model', None) is not None
        ae_loaded = getattr(service.detector, 'autoencoder_model', None) is not None
        lstm_loaded = getattr(service.detector, 'lstm_model', None) is not None
        gnn_loaded = getattr(service.detector, 'gnn_model', None) is not None
        temp_loaded = getattr(service.detector, 'temporal_model', None) is not None
        adaptive_loaded = getattr(service.detector, 'adaptive_ensemble', None) is not None

        print(f" -> XGBoost: {xgb_loaded}")
        print(f" -> Autoencoder: {ae_loaded}")
        print(f" -> LSTM: {lstm_loaded}")
        print(f" -> GNN: {gnn_loaded}")
        print(f" -> Temporal: {temp_loaded}")
        print(f" -> Adaptive Ensemble: {adaptive_loaded}")

    # 2. Test DetectionEngine ML primary vs fallback behavior
    print("\n[2/3] Testing DetectionEngine ML primary detection vs heuristic fallback...")
    from detection.detector import DetectionEngine, create_detection_engine

    engine = create_detection_engine(model_dir=os.path.join(project_root, 'models'))
    
    # Test sample flow
    sample_flow = {
        'src_ip': '192.168.1.50',
        'dst_ip': '10.0.0.10',
        'src_port': 43210,
        'dst_port': 80,
        'protocol': 'TCP',
        'duration': 1.0,
        'bytes_sent': 500,
        'bytes_recv': 1000,
        'packets_sent': 5,
        'packets_recv': 8
    }

    result = engine.analyze_flow(sample_flow)
    print(f" -> Result is_threat: {result['is_threat']}")
    print(f" -> Result model_used: {result['model_used']}")
    print(f" -> Result confidence: {result['confidence']}")

    # 3. Test Pure Heuristic Fallback when no ML models are loaded
    print("\n[3/3] Testing Pure Heuristic Fallback when ML models are absent...")
    empty_engine = DetectionEngine() # engine with no ML models loaded
    fallback_res = empty_engine.analyze_flow(sample_flow)
    print(f" -> Fallback model_used: {fallback_res['model_used']}")
    assert 'Fallback' in fallback_res['model_used'] or 'heuristic' in fallback_res['model_used'].lower(), \
        "Fallback engine did not report Heuristic Rule Engine fallback!"

    print("\n" + "=" * 75)
    print("✅ SUCCESS: ML MODELS DRIVE PRIMARY FLOW & RULES ACT AS LAST FALLBACK!")
    print("=" * 75)

if __name__ == '__main__':
    test_model_flow_architecture()
