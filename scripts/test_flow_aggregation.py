"""
Standalone Test Script for Flow Aggregation Layer
===================================================
Captures real / simulated network traffic, aggregates raw packets into
bidirectional 5-tuple network flows matching the NetworkFlow schema,
and prints exact flow metrics (duration, packets/bytes sent & recv, TCP flags).
"""

import sys
import os
import time

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collectors.live_capture import LiveCaptureManager

def main():
    print("=" * 75)
    print("🌊 AI-NIDS Flow Aggregation Layer Test")
    print("=" * 75)

    selected_iface = 'SIMULATED_TRAFFIC'
    print(f"\n1. Starting Live Packet Capture on interface '{selected_iface}'...")
    print("   Configured Flow Thresholds:")
    print("   - Idle Timeout: 2.0s")
    print("   - Active Timeout / Max Duration: 10.0s")
    print("   - Max Packets per Flow: 100")

    manager = LiveCaptureManager(detector=None)
    started = manager.start_capture(interface=selected_iface)

    if not started:
        print("❌ Failed to start live capture!")
        sys.exit(1)

    print("🟢 Live packet capture running. Aggregating flows in real-time for 6 seconds...")
    time.sleep(6)

    # Stop capture and flush remaining active flows
    print("\n2. Stopping capture & flushing remaining active flows...")
    manager.stop_capture()

    # Retrieve completed flows
    completed_flows = manager.get_recent_flows(limit=10)

    print(f"\n3. Aggregated Network Flows Count: {len(completed_flows)}")
    print("\n" + "=" * 75)
    print(f"{'FLOW 5-TUPLE':<35} | {'DUR(s)':<7} | {'PKTS(S/R)':<10} | {'BYTES(S/R)':<12} | {'FLAGS'}")
    print("=" * 75)

    for idx, flow in enumerate(completed_flows, 1):
        tuple_str = f"{flow['source_ip']}:{flow['source_port']} -> {flow['destination_ip']}:{flow['destination_port']} [{flow['protocol']}]"
        pkts_str = f"{flow['packets_sent']}/{flow['packets_recv']}"
        bytes_str = f"{flow['bytes_sent']}/{flow['bytes_recv']}"
        flags_str = f"S:{flow['syn_count']} A:{flow['ack_count']} F:{flow['fin_count']} R:{flow['rst_count']}"
        
        print(f"[{idx:02d}] {tuple_str:<30} | {flow['duration']:<7.3f} | {pkts_str:<10} | {bytes_str:<12} | {flags_str}")

    print("=" * 75)
    if len(completed_flows) > 0:
        print("✅ TEST PASSED: Packets were successfully aggregated into NetworkFlow representations!")
    else:
        print("⚠️ WARNING: No completed flows returned.")

if __name__ == '__main__':
    main()
