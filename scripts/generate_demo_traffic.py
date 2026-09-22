"""
Safe Local Traffic Generator for AI-NIDS Live Demo
===================================================
Generates benign network traffic and safe controlled high-volume connection bursts
exclusively against localhost (127.0.0.1) or local interface to trigger ML detection engine
without targeting public or external networks.

Usage:
  python scripts/generate_demo_traffic.py --type normal
  python scripts/generate_demo_traffic.py --type attack
"""

import sys
import time
import socket
import argparse
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("demo_traffic_gen")


def generate_normal_traffic(count: int = 15, delay: float = 0.5):
    logger.info(f"Generating {count} normal local HTTP/TCP requests...")
    for i in range(1, count + 1):
        try:
            # Local HTTP check or public test endpoint
            urllib.request.urlopen("http://127.0.0.1:5000/dashboard", timeout=2)
            logger.info(f"  [Normal {i}/{count}] HTTP GET 127.0.0.1:5000 -> 200 OK")
        except Exception:
            # Fallback to local socket ping
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                s.connect_ex(("127.0.0.1", 5000))
                s.close()
                logger.info(f"  [Normal {i}/{count}] TCP Connection 127.0.0.1:5000 -> OK")
            except Exception as e:
                logger.warning(f"  [Normal {i}/{count}] Connection error: {e}")
        time.sleep(delay)
    logger.info("Normal traffic generation completed.")


def generate_safe_attack_burst(count: int = 250, delay: float = 0.005):
    logger.info(f"Generating safe controlled high-rate TCP connection burst ({count} requests)...")
    success_count = 0
    start_time = time.time()
    
    for i in range(1, count + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect_ex(("127.0.0.1", 5000))
            s.close()
            success_count += 1
        except Exception:
            pass
        
        if i % 50 == 0:
            logger.info(f"  [High-Rate Burst] Dispatched {i}/{count} connection requests...")
        time.sleep(delay)

    elapsed = time.time() - start_time
    rate = count / max(elapsed, 0.001)
    logger.info(f"Safe high-rate burst completed: {count} connections in {elapsed:.2f}s ({rate:.1f} req/sec).")


def main():
    parser = argparse.ArgumentParser(description="AI-NIDS Safe Local Traffic Generator")
    parser.add_argument("--type", choices=["normal", "attack", "both"], default="both", help="Type of traffic to generate")
    parser.add_argument("--count", type=int, default=0, help="Override request count")
    args = parser.parse_args()

    print("=" * 60)
    print("AI-NIDS SAFE LOCAL TRAFFIC DEMO GENERATOR")
    print("Target: 127.0.0.1 (Localhost Only)")
    print("=" * 60)

    if args.type in ["normal", "both"]:
        c = args.count if args.count > 0 else 15
        generate_normal_traffic(count=c)

    if args.type == "both":
        logger.info("Waiting 3 seconds before high-rate burst...")
        time.sleep(3)

    if args.type in ["attack", "both"]:
        c = args.count if args.count > 0 else 250
        generate_safe_attack_burst(count=c)

    print("\n[Done] Traffic generation complete. Check AI-NIDS Live Monitoring dashboard!")


if __name__ == '__main__':
    main()
