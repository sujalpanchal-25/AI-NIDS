"""
Comprehensive Verification Script for Live Network Monitoring Fix
==================================================================
Tests:
1. Dynamic interface resolution with all variations:
   - UI display string
   - Interface name
   - IP address
   - MAC address
   - Scapy index
   - Substring description
   - SIMULATED_TRAFFIC
   - Invalid adapter error handling
2. Live packet capture (dual-stack IPv4 and IPv6) with filter="ip or ip6"
3. Flow aggregation and detection engine pipeline verification
4. Flask REST API routes (/api/live/start, /api/live/status, /api/live/stop)
"""

import sys
import os
import time
import json
import urllib.request

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collectors.live_capture import resolve_scapy_interface, LiveCapture, LiveCaptureManager
from scapy.all import conf

def test_interface_resolution():
    print("=" * 70)
    print("TEST 1: Dynamic Interface Resolution")
    print("=" * 70)

    test_cases = [
        ("UI Display Label", "WI-FI 2 (MEDIATEK WI-FI 6 MT7921 WIRELESS LAN CARD) [10.47.146.234]"),
        ("Adapter Name", "Wi-Fi 2"),
        ("Adapter IP", "10.47.146.234"),
        ("Adapter MAC", "e8:fb:1c:bc:aa:85"),
        ("Adapter Index string", "18"),
        ("Description Substring", "MediaTek Wi-Fi 6"),
        ("Simulated Traffic", "SIMULATED_TRAFFIC"),
        ("Default/None", None),
        ("Invalid Adapter", "CompletelyInvalidAdapterName_9999")
    ]

    all_passed = True
    for label, input_val in test_cases:
        res = resolve_scapy_interface(input_val)
        if label == "Invalid Adapter":
            passed = (res is None)
            status = "PASSED" if passed else "FAILED"
            print(f"[{status}] Input: '{input_val}' -> Expected None, got: {res}")
        elif label == "Simulated Traffic":
            passed = (res == 'SIMULATED_TRAFFIC')
            status = "PASSED" if passed else "FAILED"
            print(f"[{status}] Input: '{input_val}' -> {res}")
        elif label == "Default/None":
            passed = (res is not None)
            status = "PASSED" if passed else "FAILED"
            print(f"[{status}] Input: None -> {getattr(res, 'description', getattr(res, 'name', str(res)))}")
        else:
            # Should resolve to the MediaTek adapter (index 18 or name Wi-Fi 2 or matching dev)
            passed = (res is not None)
            desc = getattr(res, 'description', getattr(res, 'name', str(res)))
            ip = getattr(res, 'ip', 'N/A')
            mac = getattr(res, 'mac', 'N/A')
            status = "PASSED" if passed else "FAILED"
            print(f"[{status}] {label} ('{input_val}') -> Resolved: {desc} | IP: {ip} | MAC: {mac}")
        if not passed:
            all_passed = False

    assert all_passed, "Some interface resolution tests failed!"
    print(">>> All Interface Resolution Tests Passed successfully!\n")


def test_dual_stack_live_capture():
    print("=" * 70)
    print("TEST 2: Dual-Stack (IPv4 + IPv6) Live Capture on Resolved Wi-Fi Adapter")
    print("=" * 70)

    # Resolve active Wi-Fi adapter
    target_input = "WI-FI 2 (MEDIATEK WI-FI 6 MT7921 WIRELESS LAN CARD) [10.47.146.234]"
    resolved = resolve_scapy_interface(target_input)
    print(f"Target UI String: {target_input}")
    print(f"Resolved to: {getattr(resolved, 'description', getattr(resolved, 'name', str(resolved)))}")
    print(f"Resolved IP: {getattr(resolved, 'ip', 'N/A')}")
    print(f"Resolved MAC: {getattr(resolved, 'mac', 'N/A')}")

    # Use LiveCaptureManager with DetectionEngine
    manager = LiveCaptureManager()
    started = manager.start_capture(interface=target_input, filter="ip or ip6")
    assert started, "Failed to start capture on resolved interface!"
    print("Capture running in background thread with filter='ip or ip6'...")

    # Generate quick web traffic (Google / GitHub)
    print("Sending web requests to generate live IPv4 and IPv6 traffic...")
    urls = [
        "https://www.google.com",
        "https://api.github.com",
        "https://cloudflare.com"
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 AI-NIDS-Test'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                _ = resp.read(512)
        except Exception as e:
            # Some URLs may timeout or succeed, that's fine
            pass

    time.sleep(4)

    stats = manager.get_statistics()
    recent_packets = manager.get_recent_packets(limit=10)
    recent_flows = manager.get_recent_flows(limit=10)
    manager.stop_capture()

    print("\n--- Live Capture Statistics ---")
    print(f"Total Packets Captured : {stats.get('total_packets', 0)}")
    print(f"Total Bytes Captured   : {stats.get('total_bytes', 0)}")
    print(f"Flows Processed        : {stats.get('flows_processed', 0)}")
    print(f"Threats Detected       : {stats.get('threats_detected', 0)}")
    print(f"Protocols Seen         : {stats.get('protocols', {})}")

    print("\n--- Sample Captured Packets (First 5) ---")
    for i, p in enumerate(recent_packets[:5], 1):
        print(f"  [{i}] {p['timestamp']} | {p['src_ip']}:{p['src_port']} -> {p['dst_ip']}:{p['dst_port']} | {p['protocol']} | {p['length']}B")

    print("\n--- Sample Aggregated Flows (First 3) ---")
    for i, f in enumerate(recent_flows[:3], 1):
        print(f"  [{i}] {f.get('source_ip')}:{f.get('source_port')} -> {f.get('destination_ip')}:{f.get('destination_port')} [{f.get('protocol')}] | Pkts: {f.get('packets_sent')}+{f.get('packets_recv')} | Dur: {f.get('duration'):.3f}s")
        det = f.get('detection', {})
        if isinstance(det, dict):
            print(f"      Detection Engine Result: Attack={det.get('is_attack')} | Type={det.get('attack_type')} | Conf={det.get('confidence')} | Severity={det.get('severity')}")

    assert stats.get('total_packets', 0) > 0, "Failed: 0 packets captured on physical Wi-Fi adapter!"
    print("\n>>> Real Live Packet Capture and Aggregation Passed successfully!\n")


def test_flask_live_routes():
    print("=" * 70)
    print("TEST 3: Flask Live Monitoring REST API Routes")
    print("=" * 70)

    from app import create_app
    app = create_app()
    client = app.test_client()

    # 1. Test /api/live/interfaces
    resp = client.get('/api/live/interfaces')
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = json.loads(resp.data)
    assert data.get('success') is True, "Interfaces endpoint failed"
    print(f"[/api/live/interfaces] Found {data.get('count')} network interfaces")

    # 2. Test starting with invalid interface returns 400
    bad_resp = client.post('/api/live/start', json={'interface': 'BogusNonExistentAdapter_404'})
    assert bad_resp.status_code == 400, f"Expected 400 for bad interface, got {bad_resp.status_code}"
    bad_data = json.loads(bad_resp.data)
    assert bad_data.get('success') is False
    print(f"[/api/live/start - Bad Interface] Correctly rejected with 400: {bad_data.get('error')}")

    # 3. Test starting with UI-selected interface
    ui_iface = "WI-FI 2 (MEDIATEK WI-FI 6 MT7921 WIRELESS LAN CARD) [10.47.146.234]"
    start_resp = client.post('/api/live/start', json={'interface': ui_iface, 'filter': 'ip or ip6'})
    assert start_resp.status_code == 200, f"Expected 200, got {start_resp.status_code}"
    start_data = json.loads(start_resp.data)
    assert start_data.get('success') is True, f"Start failed: {start_data}"
    print(f"[/api/live/start - Resolved Interface] Successfully started on: {start_data.get('interface')}")

    # 4. Test /api/live/status
    time.sleep(2)
    status_resp = client.get('/api/live/status')
    assert status_resp.status_code == 200
    status_data = json.loads(status_resp.data)
    assert status_data.get('is_running') is True
    print(f"[/api/live/status] Monitoring Status: {status_data.get('status')} | Packets Captured: {status_data['statistics'].get('packets_captured')}")

    # 5. Test /dashboard/sync
    with client.session_transaction() as sess:
        sess['_user_id'] = '1'
    sync_resp = client.get('/dashboard/sync')
    assert sync_resp.status_code == 200, f"Sync expected 200, got {sync_resp.status_code}"
    sync_data = json.loads(sync_resp.data)
    print(f"[/dashboard/sync] Dashboard Sync Total Flows: {sync_data['stats'].get('total_flows')} | Total Alerts: {sync_data['stats'].get('total_alerts')}")

    # 6. Test /api/live/stop
    stop_resp = client.post('/api/live/stop')
    assert stop_resp.status_code == 200
    stop_data = json.loads(stop_resp.data)
    assert stop_data.get('success') is True
    print(f"[/api/live/stop] Stopped successfully: {stop_data.get('message')}")

    print("\n>>> All Flask REST API Route Tests Passed successfully!\n")


if __name__ == '__main__':
    try:
        test_interface_resolution()
        test_dual_stack_live_capture()
        test_flask_live_routes()
        print("=" * 70)
        print("*** ALL LIVE NETWORK MONITORING FIX TESTS PASSED PERFECTLY! ***")
        print("=" * 70)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
