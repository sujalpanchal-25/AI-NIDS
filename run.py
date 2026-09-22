#!/usr/bin/env python
"""
AI-NIDS Application Entry Point
================================
Run this file to start the Flask application.

Usage:
    python run.py                 # Development mode
    python run.py --production    # Production mode (use gunicorn instead)
"""

import os
import sys
import argparse
from pathlib import Path

# Fix Windows console UTF-8 / charmap encoding issues
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure matplotlib uses a writable local directory inside the project
mpl_dir = PROJECT_ROOT / 'data' / '.matplotlib'
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(mpl_dir))

from app import create_app
from config import get_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='AI-NIDS - AI-Powered Network Intrusion Detection System'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='Run in production mode'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set environment
    if args.production:
        os.environ['FLASK_ENV'] = 'production'
    else:
        os.environ['FLASK_ENV'] = os.environ.get('FLASK_ENV', 'development')
    
    # Create application
    app = create_app()
    
    # Print startup banner
    try:
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ██╗      ███╗   ██╗██╗██████╗ ███████╗            ║
    ║    ██╔══██╗██║      ████╗  ██║██║██╔══██╗██╔════╝            ║
    ║    ███████║██║█████╗██╔██╗ ██║██║██║  ██║███████╗            ║
    ║    ██╔══██║██║╚════╝██║╚██╗██║██║██║  ██║╚════██║            ║
    ║    ██║  ██║██║      ██║ ╚████║██║██████╔╝███████║            ║
    ╚═╝  ╚═╝╚═╝      ╚═╝  ╚═══╝╚═╝╚═════╝ ╚══════╝            ║
    ║                                                               ║
    ║    AI-Powered Network Intrusion Detection System              ║
    ║    Version 1.0.0 | SOC-Grade Security                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    except UnicodeEncodeError:
        print("""
    =================================================================
             AI-NIDS: Network Intrusion Detection System
             Version 1.0.0 | SOC-Grade Security
    =================================================================
    """)
    
    try:
        print(f"    🚀 Starting AI-NIDS...")
    except UnicodeEncodeError:
        print(f"    [*] Starting AI-NIDS...")
    print(f"    📍 Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"    🌐 URL: http://{args.host}:{args.port}")
    print(f"    📊 Dashboard: http://{args.host}:{args.port}/dashboard")
    print(f"    🔌 API: http://{args.host}:{args.port}/api/v1")
    print()
    
    # Run application
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug or (not args.production),
        threaded=True
    )


if __name__ == '__main__':
    main()
