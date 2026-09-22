"""
Live Packet Capture REST API Routes
===================================
Endpoints for listing network adapters, controlling live packet capture background threads,
and testing captured packet feeds independently of ML models.
"""

from flask import Blueprint, jsonify, request, current_app
from typing import Dict, Any, List
import logging
import os

from collectors.live_capture import LiveCapture, LiveCaptureManager

logger = logging.getLogger(__name__)

live_routes_bp = Blueprint('live_routes', __name__)

# Global singleton instance for live capture manager
# detector=None means the detection engine will be lazy-initialized on first START
_capture_manager: LiveCaptureManager = LiveCaptureManager(detector=None)


@live_routes_bp.route('/interfaces', methods=['GET'])
def get_interfaces():
    """List all available network interfaces on Windows 11 / Linux."""
    try:
        ifaces = LiveCapture.list_interfaces()
        default_iface = LiveCapture.get_default_interface()
        return jsonify({
            'success': True,
            'count': len(ifaces),
            'default_interface': default_iface,
            'interfaces': ifaces
        })
    except Exception as e:
        logger.error(f"Error listing network interfaces: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@live_routes_bp.route('/start', methods=['POST'])
def start_capture():
    """Start background packet capture on selected interface."""
    try:
        data = request.get_json(silent=True) or {}
        interface = data.get('interface')
        bpf_filter = data.get('filter', 'ip or ip6')
        if not bpf_filter or str(bpf_filter).strip() == 'ip':
            bpf_filter = 'ip or ip6'

        # Validate that selected interface can be resolved before launching capture
        resolved = LiveCapture.resolve_interface(interface)
        if resolved is None:
            err_msg = f"Could not resolve network interface '{interface}'. Please select a valid adapter."
            logger.error(err_msg)
            return jsonify({'success': False, 'error': err_msg}), 400

        if _capture_manager.is_running:
            return jsonify({
                'success': True,
                'already_running': True,
                'interface': _capture_manager.capture.interface_display_name,
                'message': f"Capture already running on interface {_capture_manager.capture.interface_display_name}"
            })

        # Lazy-initialize detection engine if not yet available.
        # This runs inside the Flask request context so app config is accessible.
        if _capture_manager.detector is None:
            try:
                from detection.detector import create_detection_engine
                # live_routes.py is at: app/routes/live_routes.py
                # Two levels up (.., ..) lands at the project root
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..', '..')
                )
                model_dir = os.path.join(project_root, 'models')
                if not os.path.isdir(model_dir):
                    # Fallback: check sibling paths
                    model_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), '..', '..', 'models')
                    )
                detector = create_detection_engine(model_dir=model_dir)
                _capture_manager.detector = detector
                _capture_manager.detection_callback.detector = detector
                logger.info(f"Detection engine lazy-initialized from: {model_dir}")
            except Exception as e_det:
                logger.warning(
                    f"Could not initialize detection engine (capture will still run without ML): {e_det}"
                )

        success = _capture_manager.start_capture(interface=interface, filter=bpf_filter)
        if success:
            return jsonify({
                'success': True,
                'interface': _capture_manager.capture.interface_display_name,
                'message': f"Packet capture started successfully on {_capture_manager.capture.interface_display_name}"
            })
        else:
            # start_capture returned False — check if there is a stored error
            err = getattr(_capture_manager.capture, '_last_capture_error', None) or \
                  'Failed to launch capture background thread'
            return jsonify({'success': False, 'error': err}), 500

    except Exception as e:
        logger.error(f"Error starting live capture: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@live_routes_bp.route('/stop', methods=['POST'])
def stop_capture():
    """Stop background packet capture and release resources."""
    try:
        if not _capture_manager.is_running:
            return jsonify({'success': True, 'message': 'Capture is not running'})
        
        _capture_manager.stop_capture()
        return jsonify({'success': True, 'message': 'Packet capture stopped successfully'})
    except Exception as e:
        logger.error(f"Error stopping live capture: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@live_routes_bp.route('/status', methods=['GET'])
def get_status():
    """Get live capture status, statistics, and any capture error."""
    try:
        stats = _capture_manager.get_statistics()
        is_running = _capture_manager.is_running

        # Surface any capture error so the frontend can show an actionable message
        # instead of silently transitioning to STOPPED.
        capture_error = getattr(_capture_manager.capture, '_last_capture_error', None)

        return jsonify({
            'success': True,
            'is_running': is_running,
            'status': 'MONITORING' if is_running else ('ERROR' if capture_error else 'STOPPED'),
            'interface': _capture_manager.capture.interface_display_name,
            'capture_error': capture_error,  # None when clean, error string when crashed
            'statistics': stats
        })
    except Exception as e:
        logger.error(f"Error fetching capture status: {e}")
        return jsonify({'success': False, 'error': str(e), 'status': 'ERROR'}), 500


@live_routes_bp.route('/recent-packets', methods=['GET'])
def get_recent_packets():
    """Retrieve recently captured raw packet headers for testing verification."""
    try:
        limit = request.args.get('limit', 50, type=int)
        packets = _capture_manager.get_recent_packets(limit=limit)
        return jsonify({
            'success': True,
            'count': len(packets),
            'packets': packets
        })
    except Exception as e:
        logger.error(f"Error fetching recent packets: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@live_routes_bp.route('/recent-flows', methods=['GET'])
def get_recent_flows():
    """Retrieve completed 5-tuple aggregated network flows for verification."""
    try:
        limit = request.args.get('limit', 50, type=int)
        flows = _capture_manager.get_recent_flows(limit=limit)
        return jsonify({
            'success': True,
            'count': len(flows),
            'flows': flows
        })
    except Exception as e:
        logger.error(f"Error fetching recent flows: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
