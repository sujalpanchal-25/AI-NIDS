"""
Real-time Network Flow Aggregation Engine for AI-NIDS
=====================================================
Groups raw captured packets into bidirectional network flows (5-tuple matching)
and calculates real flow duration, packet rates, byte counts, and TCP flags
matching the NetworkFlow database schema.
"""

import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from collectors.live_capture import CapturedPacket, PacketCallback

logger = logging.getLogger(__name__)


@dataclass
class FlowTracker:
    """Tracks a single bidirectional network flow."""
    flow_id: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    start_time: datetime
    last_time: datetime
    packets_sent: int = 0
    packets_recv: int = 0
    bytes_sent: int = 0
    bytes_recv: int = 0
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0

    def add_packet(self, packet: CapturedPacket, is_forward: bool) -> None:
        """Update flow metrics with a new captured packet."""
        self.last_time = packet.timestamp

        if is_forward:
            self.packets_sent += 1
            self.bytes_sent += packet.length
        else:
            self.packets_recv += 1
            self.bytes_recv += packet.length

        # Update TCP flags count
        if packet.flags:
            if packet.flags.get('SYN'):
                self.syn_count += 1
            if packet.flags.get('ACK'):
                self.ack_count += 1
            if packet.flags.get('FIN'):
                self.fin_count += 1
            if packet.flags.get('RST'):
                self.rst_count += 1

    @property
    def duration(self) -> float:
        """Calculate flow duration in seconds."""
        delta = (self.last_time - self.start_time).total_seconds()
        return max(0.0001, delta)

    @property
    def total_bytes(self) -> int:
        """Total bytes exchanged in flow."""
        return self.bytes_sent + self.bytes_recv

    @property
    def total_packets(self) -> int:
        """Total packets exchanged in flow."""
        return self.packets_sent + self.packets_recv

    def to_dict(self) -> Dict[str, Any]:
        """Convert flow metrics to NetworkFlow dictionary matching DB schema."""
        return {
            'timestamp': self.start_time,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'source_port': self.source_port,
            'destination_port': self.destination_port,
            'protocol': self.protocol,
            'duration': round(self.duration, 4),
            'total_bytes': self.total_bytes,
            'bytes_sent': self.bytes_sent,
            'bytes_recv': self.bytes_recv,
            'packets_sent': self.packets_sent,
            'packets_recv': self.packets_recv,
            'total_packets': self.total_packets,
            'syn_count': self.syn_count,
            'ack_count': self.ack_count,
            'fin_count': self.fin_count,
            'rst_count': self.rst_count,
            'flow_id': self.flow_id
        }


class FlowAggregator(PacketCallback):
    """
    Bidirectional Flow Aggregator.
    Collects raw packets, groups them by 5-tuple, and flushes completed flows
    based on idle timeout, max duration, or packet limit.
    """

    def __init__(
        self,
        idle_timeout: float = 2.0,
        active_timeout: float = 10.0,
        max_packets_per_flow: int = 100,
        flow_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize flow aggregator.
        
        Args:
            idle_timeout: Seconds of inactivity before closing a flow (default 2.0s)
            active_timeout: Max flow duration before forcing flush (default 10.0s)
            max_packets_per_flow: Packet threshold before flushing flow
            flow_callback: Optional callback function called whenever a flow completes
        """
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.max_packets_per_flow = max_packets_per_flow
        self.flow_callback = flow_callback

        # Flow tables and concurrency lock
        self._active_flows: Dict[str, FlowTracker] = {}
        self._lock = threading.Lock()
        self.flow_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.total_flows_count: int = 0
        from collections import deque
        self._completed_history: deque = deque(maxlen=2000)

        # Cleaner background thread
        self._stop_event = threading.Event()
        self._cleaner_thread: Optional[threading.Thread] = None

    def _get_flow_keys(self, packet: CapturedPacket) -> Tuple[str, str]:
        """Generate forward and reverse 5-tuple flow keys."""
        fwd_key = f"{packet.src_ip}:{packet.src_port}->{packet.dst_ip}:{packet.dst_port}-{packet.protocol}"
        rev_key = f"{packet.dst_ip}:{packet.dst_port}->{packet.src_ip}:{packet.src_port}-{packet.protocol}"
        return fwd_key, rev_key

    def on_packet(self, packet: CapturedPacket) -> None:
        """Process incoming captured packet into flow aggregations."""
        fwd_key, rev_key = self._get_flow_keys(packet)
        flow_to_emit: Optional[Dict[str, Any]] = None

        with self._lock:
            # Check if forward flow exists
            if fwd_key in self._active_flows:
                flow = self._active_flows[fwd_key]
                flow.add_packet(packet, is_forward=True)
                if self._should_flush(flow):
                    flow_to_emit = self._active_flows.pop(fwd_key).to_dict()

            # Check if reverse flow exists
            elif rev_key in self._active_flows:
                flow = self._active_flows[rev_key]
                flow.add_packet(packet, is_forward=False)
                if self._should_flush(flow):
                    flow_to_emit = self._active_flows.pop(rev_key).to_dict()

            # Create new flow
            else:
                flow = FlowTracker(
                    flow_id=fwd_key,
                    source_ip=packet.src_ip,
                    destination_ip=packet.dst_ip,
                    source_port=packet.src_port,
                    destination_port=packet.dst_port,
                    protocol=packet.protocol,
                    start_time=packet.timestamp,
                    last_time=packet.timestamp
                )
                flow.add_packet(packet, is_forward=True)
                
                if self._should_flush(flow):
                    flow_to_emit = flow.to_dict()
                else:
                    self._active_flows[fwd_key] = flow

        # Emit completed flow outside lock
        if flow_to_emit:
            self._emit_flow(flow_to_emit)

    def _should_flush(self, flow: FlowTracker) -> bool:
        """Check if flow has reached active timeout, packet limit, or TCP termination."""
        if flow.total_packets >= self.max_packets_per_flow:
            return True
        if flow.duration >= self.active_timeout:
            return True
        if flow.fin_count > 0 or flow.rst_count > 0:
            return True
        return False

    def _emit_flow(self, flow_dict: Dict[str, Any]) -> None:
        """Put completed flow into queue and invoke callback."""
        self.total_flows_count += 1
        if hasattr(self, '_completed_history'):
            self._completed_history.append(flow_dict)

        try:
            self.flow_queue.put_nowait(flow_dict)
        except queue.Full:
            try:
                self.flow_queue.get_nowait()
                self.flow_queue.put_nowait(flow_dict)
            except queue.Empty:
                pass

        if self.flow_callback:
            try:
                self.flow_callback(flow_dict)
            except Exception as e:
                logger.error(f"Flow callback error: {e}")

    def _cleanup_loop(self) -> None:
        """Background thread loop flushing idle expired flows."""
        while not self._stop_event.is_set():
            time.sleep(1.0)
            now = datetime.now()
            expired_flows: List[Dict[str, Any]] = []

            with self._lock:
                for key, flow in list(self._active_flows.items()):
                    idle_secs = (now - flow.last_time).total_seconds()
                    if idle_secs >= self.idle_timeout or flow.duration >= self.active_timeout:
                        expired_flows.append(self._active_flows.pop(key).to_dict())

            for flow_dict in expired_flows:
                self._emit_flow(flow_dict)

    def on_start(self) -> None:
        """Called when capture starts."""
        self._stop_event.clear()
        with self._lock:
            self._active_flows.clear()

        # Start cleanup thread
        self._cleaner_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleaner_thread.start()
        logger.info("FlowAggregator started background cleanup thread")

    def on_stop(self) -> None:
        """Called when capture stops — flush all remaining active flows."""
        self._stop_event.set()
        if self._cleaner_thread and self._cleaner_thread.is_alive():
            self._cleaner_thread.join(timeout=2.0)

        remaining_flows: List[Dict[str, Any]] = []
        with self._lock:
            for key, flow in list(self._active_flows.items()):
                remaining_flows.append(flow.to_dict())
            self._active_flows.clear()

        for flow_dict in remaining_flows:
            self._emit_flow(flow_dict)

        logger.info(f"FlowAggregator stopped. Flushed {len(remaining_flows)} remaining flows.")

    def get_completed_flows(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve completed flows without losing historical counts."""
        if hasattr(self, '_completed_history') and self._completed_history:
            return list(self._completed_history)[-limit:]

        flows = []
        while not self.flow_queue.empty() and len(flows) < limit:
            try:
                flows.append(self.flow_queue.get_nowait())
            except queue.Empty:
                break
        return flows
