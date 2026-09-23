"""
Live Network Capture Module for AI-NIDS
Real-time packet capture and analysis using Scapy.
Requires: Npcap (Windows) or libpcap (Linux/Mac)
"""

import os
import sys
import re
import json
import logging
import threading
import queue
import time
import random  # Used only by the SIMULATED_TRAFFIC demo generator
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class CapturedPacket:
    """Represents a captured network packet."""
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    length: int
    flags: Dict[str, bool] = field(default_factory=dict)
    payload: bytes = b''
    raw: bytes = b''
    interface: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'length': self.length,
            'flags': self.flags,
            'interface': self.interface
        }


class PacketCallback(ABC):
    """Abstract base class for packet callbacks."""
    
    @abstractmethod
    def on_packet(self, packet: CapturedPacket) -> None:
        """Called for each captured packet."""
        pass
    
    def on_start(self) -> None:
        """Called when capture starts."""
        pass
    
    def on_stop(self) -> None:
        """Called when capture stops."""
        pass


class PrintCallback(PacketCallback):
    """Simple callback that prints packet info."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.count = 0
    
    def on_packet(self, packet: CapturedPacket) -> None:
        self.count += 1
        if self.verbose:
            print(f"[{self.count}] {packet.timestamp.strftime('%H:%M:%S.%f')[:-3]} "
                  f"{packet.src_ip}:{packet.src_port} -> {packet.dst_ip}:{packet.dst_port} "
                  f"[{packet.protocol}] {packet.length} bytes")
        else:
            if self.count % 100 == 0:
                print(f"Captured {self.count} packets...")
    
    def on_start(self) -> None:
        print("Starting capture...")
        self.count = 0
    
    def on_stop(self) -> None:
        print(f"Capture stopped. Total packets: {self.count}")


class QueueCallback(PacketCallback):
    """Callback that puts packets into a queue for processing."""
    
    def __init__(self, max_size: int = 10000):
        self.packet_queue: queue.Queue = queue.Queue(maxsize=max_size)
    
    def on_packet(self, packet: CapturedPacket) -> None:
        try:
            self.packet_queue.put_nowait(packet)
        except queue.Full:
            # Drop oldest packet
            try:
                self.packet_queue.get_nowait()
                self.packet_queue.put_nowait(packet)
            except queue.Empty:
                pass
    
    def get_packet(self, timeout: float = 1.0) -> Optional[CapturedPacket]:
        try:
            return self.packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_packets(self) -> List[CapturedPacket]:
        packets = []
        while not self.packet_queue.empty():
            try:
                packets.append(self.packet_queue.get_nowait())
            except queue.Empty:
                break
        return packets


class StatisticsCallback(PacketCallback):
    """Callback that collects traffic statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_packets = 0
        self.total_bytes = 0
        self.protocols: Dict[str, int] = defaultdict(int)
        self.src_ips: Dict[str, int] = defaultdict(int)
        self.dst_ips: Dict[str, int] = defaultdict(int)
        self.ports: Dict[int, int] = defaultdict(int)
        self.start_time: Optional[datetime] = None
        self.last_time: Optional[datetime] = None
    
    def on_packet(self, packet: CapturedPacket) -> None:
        self.total_packets += 1
        self.total_bytes += packet.length
        self.protocols[packet.protocol] += 1
        self.src_ips[packet.src_ip] += 1
        self.dst_ips[packet.dst_ip] += 1
        
        if packet.src_port:
            self.ports[packet.src_port] += 1
        if packet.dst_port:
            self.ports[packet.dst_port] += 1
        
        if self.start_time is None:
            self.start_time = packet.timestamp
        self.last_time = packet.timestamp
    
    def on_start(self) -> None:
        self.reset()
    
    def get_statistics(self) -> Dict[str, Any]:
        duration = 0
        if self.start_time and self.last_time:
            duration = (self.last_time - self.start_time).total_seconds()
        
        return {
            'total_packets': self.total_packets,
            'total_bytes': self.total_bytes,
            'duration_seconds': duration,
            'packets_per_second': self.total_packets / max(duration, 0.001),
            'bytes_per_second': self.total_bytes / max(duration, 0.001),
            'protocols': dict(self.protocols),
            'top_source_ips': dict(sorted(self.src_ips.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_destination_ips': dict(sorted(self.dst_ips.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_ports': dict(sorted(self.ports.items(), key=lambda x: x[1], reverse=True)[:10])
        }


class DetectionCallback(PacketCallback):
    """
    Callback that performs real-time anomaly detection.
    Integrates with the AI-NIDS detection engine.
    """
    
    def __init__(self, detector=None, alert_callback: Optional[Callable] = None):
        self.detector = detector
        self.alert_callback = alert_callback
        self.flow_buffer: Dict[str, List[CapturedPacket]] = defaultdict(list)
        self.buffer_size = 100
        self.alerts: List[Dict] = []
    
    def _get_flow_key(self, packet: CapturedPacket) -> str:
        """Generate bidirectional flow key."""
        ips = tuple(sorted([packet.src_ip, packet.dst_ip]))
        ports = tuple(sorted([packet.src_port, packet.dst_port]))
        return f"{ips[0]}:{ports[0]}-{ips[1]}:{ports[1]}-{packet.protocol}"
    
    def on_packet(self, packet: CapturedPacket) -> None:
        flow_key = self._get_flow_key(packet)
        self.flow_buffer[flow_key].append(packet)
        
        # Process when buffer is full
        if len(self.flow_buffer[flow_key]) >= self.buffer_size:
            self._process_flow(flow_key)
    
    def _process_flow(self, flow_key: str) -> None:
        """Process buffered packets for a flow."""
        packets = self.flow_buffer[flow_key]
        
        if not packets or not self.detector:
            self.flow_buffer[flow_key] = []
            return
        
        # Extract features
        features = self._extract_features(packets)
        
        # Run detection
        try:
            if hasattr(self.detector, 'detect'):
                det_res = self.detector.detect(features)
                if isinstance(det_res, list):
                    det_res = det_res[0]
                is_attack = getattr(det_res, 'is_attack', False)
                attack_type = getattr(det_res, 'attack_type', 'Normal')
                confidence = getattr(det_res, 'confidence', 0.0)
                
                if is_attack:
                    alert = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'flow_key': flow_key,
                        'risk_score': round(confidence, 4),
                        'attack_type': attack_type,
                        'details': det_res.to_dict() if hasattr(det_res, 'to_dict') else str(det_res)
                    }
                    self.alerts.append(alert)
                    if self.alert_callback:
                        self.alert_callback(alert)
            elif hasattr(self.detector, 'analyze_features'):
                result = self.detector.analyze_features(features)
                if result and result.get('is_anomaly', False):
                    alert = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'flow_key': flow_key,
                        'risk_score': result.get('risk_score', 0),
                        'attack_type': result.get('attack_type', 'Unknown'),
                        'details': result
                    }
                    self.alerts.append(alert)
                    if self.alert_callback:
                        self.alert_callback(alert)
                    
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        # Clear buffer
        self.flow_buffer[flow_key] = []
    
    def _extract_features(self, packets: List[CapturedPacket]) -> Dict[str, Any]:
        """Extract ML features from packet list."""
        import numpy as np
        
        if not packets:
            return {}
        
        lengths = [p.length for p in packets]
        
        # Calculate inter-arrival times
        iats = []
        for i in range(1, len(packets)):
            iat = (packets[i].timestamp - packets[i-1].timestamp).total_seconds()
            iats.append(iat)
        iats = iats if iats else [0]
        
        # Count directions & byte totals
        first_src = packets[0].src_ip
        bytes_sent = sum(p.length for p in packets if p.src_ip == first_src)
        bytes_recv = sum(p.length for p in packets if p.src_ip != first_src)
        packets_sent = sum(1 for p in packets if p.src_ip == first_src)
        packets_recv = len(packets) - packets_sent
        
        # Count flags
        flags = defaultdict(int)
        for p in packets:
            for flag, val in p.flags.items():
                if val:
                    flags[flag] += 1
        
        duration = (packets[-1].timestamp - packets[0].timestamp).total_seconds()
        
        return {
            'timestamp': packets[0].timestamp,
            'source_ip': packets[0].src_ip,
            'destination_ip': packets[0].dst_ip,
            'source_port': packets[0].src_port,
            'destination_port': packets[0].dst_port,
            'src_ip': packets[0].src_ip,
            'dst_ip': packets[0].dst_ip,
            'src_port': packets[0].src_port,
            'dst_port': packets[0].dst_port,
            'duration': duration,
            'protocol': packets[0].protocol,
            'packets_sent': packets_sent,
            'packets_recv': packets_recv,
            'bytes_sent': bytes_sent,
            'bytes_recv': bytes_recv,
            'total_packets': len(packets),
            'packets_forward': packets_sent,
            'packets_backward': packets_recv,
            'total_bytes': sum(lengths),
            'packet_length_mean': float(np.mean(lengths)),
            'packet_length_std': float(np.std(lengths)),
            'packet_length_min': min(lengths),
            'packet_length_max': max(lengths),
            'iat_mean': float(np.mean(iats)),
            'iat_std': float(np.std(iats)),
            'syn_count': flags.get('SYN', 0),
            'ack_count': flags.get('ACK', 0),
            'fin_count': flags.get('FIN', 0),
            'rst_count': flags.get('RST', 0)
        }
    
    def on_stop(self) -> None:
        # Process remaining flows
        for flow_key in list(self.flow_buffer.keys()):
            if self.flow_buffer[flow_key]:
                self._process_flow(flow_key)


def resolve_scapy_interface(selected_interface: Any) -> Any:
    """
    Robustly resolves a UI-selected or user-provided network interface identifier
    to an actual Scapy interface object or driver name usable by scapy.sniff().

    Accepts:
    - Scapy Interface objects directly
    - 'SIMULATED_TRAFFIC'
    - None / '' / 'default' / 'DEFAULT' / 'Default Interface' -> conf.iface
    - Numeric Scapy index (e.g., 18 or '18')
    - Scapy interface names (e.g., 'Wi-Fi 2', 'Ethernet')
    - Network adapter descriptions (e.g., 'MediaTek Wi-Fi 6 MT7921 Wireless LAN Card')
    - IPv4 addresses (e.g., '10.47.146.234')
    - IPv6 addresses (e.g., 'fe80::...')
    - MAC addresses (e.g., 'e8:fb:1c:bc:aa:85' or 'e8-fb-1c-bc-aa-85')
    - Device GUID / network_name (e.g., '\\Device\\NPF_{CDCEA133-77F6-4FEA-81BA-785FA0A63B3A}')
    - Compound UI display strings (e.g., 'WI-FI 2 (MEDIATEK WI-FI 6 MT7921 WIRELESS LAN CARD) [10.47.146.234]')

    Returns:
        The matched Scapy NetworkInterface object or 'SIMULATED_TRAFFIC', or None if unresolved.
    """
    if not selected_interface or str(selected_interface).strip().lower() in ('default', 'none', ''):
        try:
            from scapy.all import conf
            return getattr(conf, 'iface', None)
        except Exception:
            return None

    target_str = str(selected_interface).strip()
    if target_str.upper() == 'SIMULATED_TRAFFIC':
        return 'SIMULATED_TRAFFIC'

    try:
        from scapy.all import conf
    except ImportError:
        return target_str

    # 1. If it's already a Scapy interface object
    if hasattr(selected_interface, 'name') and hasattr(selected_interface, 'network_name'):
        return selected_interface

    # 2. Check numeric index (e.g., 18 or '18')
    if target_str.isdigit():
        try:
            idx = int(target_str)
            if hasattr(conf, 'ifaces') and hasattr(conf.ifaces, 'dev_from_index'):
                dev = conf.ifaces.dev_from_index(idx)
                if dev:
                    return dev
        except Exception:
            pass

    # Extract IPv4 address if present in brackets or text
    ip_match = re.search(r'\[?\b((?:\d{1,3}\.){3}\d{1,3})\b\]?', target_str)
    target_ip = ip_match.group(1) if ip_match else None

    # Extract MAC address if present
    mac_match = re.search(r'([0-9a-fA-F]{2}[:-]){5}([0-9a-fA-F]{2})', target_str)
    target_mac = mac_match.group(0).lower().replace('-', ':') if mac_match else None

    candidates = list(getattr(conf, 'ifaces', {}).values()) if hasattr(conf, 'ifaces') else []

    # 3. Match by exact IP address
    if target_ip:
        for dev in candidates:
            if getattr(dev, 'ip', None) == target_ip:
                return dev

    # 4. Match by exact MAC address
    if target_mac:
        for dev in candidates:
            dev_mac = str(getattr(dev, 'mac', '')).lower().replace('-', ':')
            if dev_mac == target_mac:
                return dev

    target_lower = target_str.lower()

    # 5. Exact match on name, description, network_name, guid, or str(dev)
    for dev in candidates:
        name = str(getattr(dev, 'name', '')).lower()
        desc = str(getattr(dev, 'description', '')).lower()
        net_name = str(getattr(dev, 'network_name', '')).lower()
        guid = str(getattr(dev, 'guid', '')).lower()
        if target_lower in (name, desc, net_name, guid, str(dev).lower()):
            return dev

    # 6. Compound label or substring match (e.g. 'Wi-Fi 2 (MediaTek Wi-Fi 6...)')
    for dev in candidates:
        name = str(getattr(dev, 'name', '')).lower()
        desc = str(getattr(dev, 'description', '')).lower()
        net_name = str(getattr(dev, 'network_name', '')).lower()
        if name and (name in target_lower or target_lower in name):
            return dev
        if desc and (desc in target_lower or target_lower in desc):
            return dev
        if net_name and (net_name in target_lower or target_lower in net_name):
            return dev

    # 7. Check if target matches get_if_list directly
    try:
        from scapy.all import get_if_list
        if target_str in get_if_list():
            return target_str
    except Exception:
        pass

    return None


class LiveCapture:
    """
    Real-time network packet capture using Scapy.
    
    Features:
    - Multi-interface support with robust Windows/Linux adapter resolution
    - Dual-stack IPv4 & IPv6 packet capture with BPF filtering
    - Callback-based packet processing
    - Asynchronous capture with threading
    - Graceful start/stop
    
    Requirements:
    - Windows: Npcap (https://npcap.com/)
    - Linux: libpcap (usually pre-installed)
    - Mac: libpcap (pre-installed)
    """
    
    PROTOCOL_MAP = {
        1: 'ICMP',
        6: 'TCP',
        17: 'UDP',
        47: 'GRE',
        58: 'ICMPv6'
    }
    
    def __init__(self):
        self._scapy = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: List[PacketCallback] = []
        self._is_capturing = False
        self.interface: Any = None
        self.selected_interface_name: str = ''
        self.filter: Optional[str] = None
        self._packet_count: int = 0
        # Stores the last capture error message (None = no error / clean stop)
        self._last_capture_error: Optional[str] = None
        
        # Load Scapy
        self._load_scapy()
    
    def _load_scapy(self):
        """Load Scapy library."""
        try:
            from scapy import all as scapy
            self._scapy = scapy
            
            # Suppress Scapy warnings
            logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
            
        except ImportError:
            raise ImportError(
                "Scapy is required for live capture. "
                "Install with: pip install scapy\n"
                "Windows users also need Npcap: https://npcap.com/"
            )
    
    @staticmethod
    def list_interfaces() -> List[Dict[str, Any]]:
        """List available network interfaces on Windows 11 / Linux."""
        interfaces = []
        seen_names = set()

        # Get default interface name string
        default_name = LiveCapture.get_default_interface()

        # 1. Scapy Interface Discovery (conf.ifaces)
        try:
            from scapy.all import conf
            if hasattr(conf, 'ifaces'):
                for dev_id, dev in conf.ifaces.items():
                    name = getattr(dev, 'name', str(dev_id))
                    desc = getattr(dev, 'description', name)
                    ip_addr = getattr(dev, 'ip', '')
                    mac_addr = getattr(dev, 'mac', 'N/A')
                    idx = getattr(dev, 'index', None)
                    net_name = getattr(dev, 'network_name', getattr(dev, 'pcap_name', ''))

                    if name and name not in seen_names:
                        is_def = (name == default_name or desc == default_name or str(dev_id) == default_name)
                        interfaces.append({
                            'name': name,
                            'description': desc if desc != name else '',
                            'ip': ip_addr if ip_addr and ip_addr != '0.0.0.0' else '',
                            'mac': mac_addr,
                            'index': idx,
                            'network_name': net_name,
                            'is_default': is_def,
                            'type': 'Network Adapter'
                        })
                        seen_names.add(name)
        except Exception as e_scapy_ifaces:
            logger.debug(f"Scapy conf.ifaces error: {e_scapy_ifaces}")

        # 2. Try psutil for human-readable Windows adapter names
        try:
            import psutil
            addrs = psutil.net_if_addrs()
            for iface_name, net_addrs in addrs.items():
                if iface_name not in seen_names:
                    ip_val = ''
                    mac_val = 'N/A'
                    for addr in net_addrs:
                        if getattr(addr, 'family', None) and str(addr.family).endswith('AF_INET'):
                            ip_val = addr.address
                        elif getattr(addr, 'family', None) and 'AF_LINK' in str(addr.family):
                            mac_val = addr.address

                    is_def = (iface_name == default_name)
                    interfaces.append({
                        'name': iface_name,
                        'description': 'System Adapter',
                        'ip': ip_val,
                        'mac': mac_val,
                        'index': None,
                        'is_default': is_def,
                        'type': 'Physical/Virtual Adapter'
                    })
                    seen_names.add(iface_name)
        except Exception as e_psutil:
            logger.debug(f"psutil list_interfaces error: {e_psutil}")

        # 3. Try Scapy get_if_list fallback
        try:
            from scapy.all import get_if_list
            for iface_name in get_if_list():
                if iface_name not in seen_names:
                    interfaces.append({
                        'name': iface_name,
                        'description': '',
                        'ip': '',
                        'mac': 'N/A',
                        'index': None,
                        'is_default': (iface_name == default_name),
                        'type': 'Scapy Interface'
                    })
                    seen_names.add(iface_name)
        except Exception as e_get_if_list:
            logger.debug(f"Scapy get_if_list error: {e_get_if_list}")

        # 4. Always include SIMULATED test driver option for non-admin / Npcap-less testing
        if 'SIMULATED_TRAFFIC' not in seen_names:
            interfaces.append({
                'name': 'SIMULATED_TRAFFIC',
                'description': 'Simulated Traffic Generator (Safe Demo)',
                'ip': '127.0.0.1',
                'mac': '00:11:22:33:44:55',
                'index': -1,
                'is_default': False,
                'type': 'Simulated Test Driver'
            })

        return interfaces
    
    @staticmethod
    def resolve_interface(selected_interface: Any) -> Any:
        """Resolve interface identifier using resolve_scapy_interface."""
        return resolve_scapy_interface(selected_interface)

    @property
    def interface_display_name(self) -> str:
        """Human-readable description of current capturing interface."""
        if self.interface == 'SIMULATED_TRAFFIC':
            return 'SIMULATED_TRAFFIC (Safe Generator)'
        if hasattr(self.interface, 'description') and self.interface.description:
            name = getattr(self.interface, 'name', '')
            desc = self.interface.description
            ip = getattr(self.interface, 'ip', '')
            label = f"{name} ({desc})" if desc != name else name
            if ip:
                label += f" [{ip}]"
            return label
        if hasattr(self.interface, 'name') and self.interface.name:
            return self.interface.name
        return str(self.interface or 'Default Interface')

    @staticmethod
    def get_default_interface() -> str:
        """Get the default network interface name string."""
        try:
            from scapy.all import conf
            iface = conf.iface
            if hasattr(iface, 'name') and iface.name:
                return str(iface.name)
            elif hasattr(iface, 'description') and iface.description:
                return str(iface.description)
            return str(iface)
        except Exception:
            return "Default Interface"
    
    def add_callback(self, callback: PacketCallback) -> None:
        """Add a packet processing callback."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: PacketCallback) -> None:
        """Remove a packet processing callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _parse_packet(self, scapy_pkt) -> Optional[CapturedPacket]:
        """Parse Scapy packet into CapturedPacket."""
        try:
            scapy = self._scapy
            
            # Check for IP layer
            if scapy.IP in scapy_pkt:
                ip = scapy_pkt[scapy.IP]
                src_ip = ip.src
                dst_ip = ip.dst
                proto_num = ip.proto
            elif scapy.IPv6 in scapy_pkt:
                ip = scapy_pkt[scapy.IPv6]
                src_ip = ip.src
                dst_ip = ip.dst
                proto_num = ip.nh
            else:
                return None
            
            protocol = self.PROTOCOL_MAP.get(proto_num, f'OTHER({proto_num})')
            
            # Get transport layer
            src_port = 0
            dst_port = 0
            flags = {}
            
            if scapy.TCP in scapy_pkt:
                tcp = scapy_pkt[scapy.TCP]
                src_port = tcp.sport
                dst_port = tcp.dport
                protocol = 'TCP'
                flag_str = str(tcp.flags) if hasattr(tcp, 'flags') else ''
                flags = {
                    'SYN': 'S' in flag_str,
                    'ACK': 'A' in flag_str,
                    'FIN': 'F' in flag_str,
                    'RST': 'R' in flag_str,
                    'PSH': 'P' in flag_str,
                    'URG': 'U' in flag_str
                }
            elif scapy.UDP in scapy_pkt:
                udp = scapy_pkt[scapy.UDP]
                src_port = udp.sport
                dst_port = udp.dport
                protocol = 'UDP'
            elif scapy.ICMP in scapy_pkt:
                protocol = 'ICMP'
            elif hasattr(scapy, 'ICMPv6EchoRequest') and (scapy.ICMPv6EchoRequest in scapy_pkt or scapy.ICMPv6EchoReply in scapy_pkt):
                protocol = 'ICMPv6'
            
            # Security/Privacy: Do NOT store raw packet payload or raw bytes permanently
            return CapturedPacket(
                timestamp=datetime.fromtimestamp(float(scapy_pkt.time)),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                length=len(scapy_pkt),
                flags=flags,
                payload=b'',  # Stripped for privacy/security
                raw=b'',      # Stripped for privacy/security
                interface=getattr(self.interface, 'name', str(self.interface or ''))
            )
            
        except Exception as e:
            logger.debug(f"Error parsing packet: {e}")
            return None
    
    def _packet_handler(self, scapy_pkt):
        """Handle captured packet."""
        if self._stop_event.is_set():
            return
        
        packet = self._parse_packet(scapy_pkt)
        if packet:
            self._packet_count += 1
            if self._packet_count <= 15 or self._packet_count % 100 == 0:
                logger.info(
                    f"PACKET CAPTURED [#{self._packet_count}]:\n"
                    f"{packet.src_ip}:{packet.src_port} -> {packet.dst_ip}:{packet.dst_port}\n"
                    f"Protocol: {packet.protocol}"
                )
            for callback in self._callbacks:
                try:
                    callback.on_packet(packet)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def _capture_loop(self, count: int, timeout: Optional[int]):
        """Main capture loop running in background thread."""
        scapy = self._scapy
        capture_crashed = False  # Distinguishes crash from intentional stop

        try:
            # Notify callbacks
            for callback in self._callbacks:
                callback.on_start()

            if self.interface == 'SIMULATED_TRAFFIC':
                # ----------------------------------------------------------------
                # DEMO / TEST PATH — only runs when user explicitly selects
                # 'SIMULATED_TRAFFIC'.  Never used as a fallback for real capture.
                # ----------------------------------------------------------------
                logger.info("Running SIMULATED_TRAFFIC demo generator (explicit demo mode).")
                packet_count = 0
                while not self._stop_event.is_set():
                    packet_count += 1
                    sim_pkt = CapturedPacket(
                        timestamp=datetime.utcnow(),
                        src_ip=f"192.168.1.{(packet_count % 50) + 1}",
                        dst_ip=f"10.0.0.{(packet_count % 20) + 1}",
                        src_port=random.randint(1024, 65535),
                        dst_port=random.choice([80, 443, 22, 53, 8080]),
                        protocol=random.choice(['TCP', 'UDP', 'ICMP']),
                        length=random.randint(64, 1500),
                        flags={'SYN': random.random() < 0.2, 'ACK': True},
                        payload=b'',
                        raw=b'',
                        interface='SIMULATED_TRAFFIC'
                    )
                    for callback in self._callbacks:
                        try:
                            callback.on_packet(sim_pkt)
                        except Exception as e_cb:
                            logger.error(f"Simulated callback error: {e_cb}")

                    time.sleep(0.1)  # 10 packets per second
                    if count > 0 and packet_count >= count:
                        break
            else:
                # ----------------------------------------------------------------
                # REAL CAPTURE PATH — Scapy/Npcap on the actual network adapter.
                # If this raises, we record the error and surface it to the UI.
                # We NEVER fall back to simulated traffic.
                # ----------------------------------------------------------------
                iface_desc = getattr(
                    self.interface, 'description',
                    getattr(self.interface, 'name', str(self.interface))
                )
                iface_ip  = getattr(self.interface, 'ip',  'N/A')
                iface_mac = getattr(self.interface, 'mac', 'N/A')
                logger.info(
                    f"CAPTURE STARTED\n"
                    f"  Interface : {iface_desc}\n"
                    f"  IPv4      : {iface_ip}\n"
                    f"  MAC       : {iface_mac}\n"
                    f"  Filter    : {self.filter}"
                )
                try:
                    scapy.sniff(
                        iface=self.interface,
                        filter=self.filter,
                        prn=self._packet_handler,
                        count=count if count > 0 else 0,
                        timeout=timeout,
                        stop_filter=lambda x: self._stop_event.is_set(),
                        store=False
                    )
                except Exception as sniff_err:
                    # Real capture failed — surface the error, do NOT fake traffic.
                    capture_crashed = True
                    err_detail = (
                        f"Packet capture failed on interface '{iface_desc}' "
                        f"(IP: {iface_ip}): {sniff_err}\n"
                        "Common causes: Npcap not installed, insufficient privileges, "
                        "interface disappeared."
                    )
                    self._last_capture_error = err_detail
                    logger.error(f"REAL CAPTURE ERROR: {err_detail}")

        except Exception as e:
            # Unexpected error before sniff even started (e.g. callback.on_start crash)
            capture_crashed = True
            err_detail = f"Capture initialization error: {e}"
            self._last_capture_error = err_detail
            logger.error(err_detail)

        finally:
            self._is_capturing = False
            # Notify callbacks — always flush flows cleanly
            for callback in self._callbacks:
                try:
                    callback.on_stop()
                except Exception as e_stop:
                    logger.error(f"Callback on_stop error: {e_stop}")

            if capture_crashed:
                logger.error(
                    "Live capture STOPPED due to error. "
                    "No fake/simulated traffic will be generated. "
                    f"Error: {self._last_capture_error}"
                )
            elif self._stop_event.is_set():
                logger.info("Live capture stopped cleanly (user requested stop).")
            else:
                logger.info("Live capture completed (count/timeout reached).")
    
    def start(
        self,
        interface: Optional[str] = None,
        filter: Optional[str] = None,
        count: int = 0,
        timeout: Optional[int] = None,
        async_capture: bool = True
    ) -> None:
        """
        Start packet capture.
        
        Args:
            interface: Network interface (None for default)
            filter: BPF filter string (defaults to "ip or ip6")
            count: Number of packets to capture (0 for unlimited)
            timeout: Capture timeout in seconds (None for unlimited)
            async_capture: Run capture in background thread
        """
        if self._is_capturing:
            logger.warning("Capture already running")
            return

        # Resolve selected interface
        resolved = resolve_scapy_interface(interface)
        if resolved is None:
            err_msg = f"Could not resolve network interface: '{interface}'. Please select a valid adapter."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.interface = resolved
        self.selected_interface_name = str(interface or 'DEFAULT')
        self._packet_count = 0
        # Clear any previous capture error on a fresh start
        self._last_capture_error = None

        # Normalize filter: default to dual-stack IPv4 + IPv6
        if not filter or str(filter).strip() == 'ip':
            filter = 'ip or ip6'
        self.filter = filter

        # Diagnostic log details
        if resolved == 'SIMULATED_TRAFFIC':
            resolved_desc = 'SIMULATED_TRAFFIC (Safe Generator)'
            resolved_ip = '127.0.0.1'
            resolved_mac = '00:11:22:33:44:55'
        else:
            resolved_desc = getattr(resolved, 'description', getattr(resolved, 'name', str(resolved)))
            resolved_ip = getattr(resolved, 'ip', 'N/A')
            resolved_mac = getattr(resolved, 'mac', 'N/A')

        logger.info(f"Selected interface from UI: {self.selected_interface_name}")
        logger.info(f"Resolved Scapy interface: {resolved_desc}")
        logger.info(f"Resolved interface IP: {resolved_ip}")
        logger.info(f"Resolved interface MAC: {resolved_mac}")
        logger.info(f"Capture filter: {self.filter}")
        logger.info("Starting live packet capture...")

        self._stop_event.clear()
        self._is_capturing = True

        if async_capture:
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(count, timeout),
                daemon=True
            )
            self._capture_thread.start()
        else:
            self._capture_loop(count, timeout)
    
    def stop(self, wait: bool = True, timeout: float = 5.0) -> None:
        """
        Stop packet capture.
        
        Args:
            wait: Wait for capture thread to finish
            timeout: Maximum time to wait
        """
        if not self._is_capturing:
            return
        
        logger.info("Stopping capture...")
        self._stop_event.set()
        
        if wait and self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
    
    @property
    def is_capturing(self) -> bool:
        """Check if capture is currently running."""
        return self._is_capturing
    
    def capture_to_file(
        self,
        filepath: str,
        interface: Optional[str] = None,
        filter: Optional[str] = None,
        count: int = 0,
        timeout: Optional[int] = None
    ) -> str:
        """
        Capture packets and save to PCAP file.
        
        Args:
            filepath: Output PCAP file path
            interface: Network interface
            filter: BPF filter
            count: Packet count
            timeout: Timeout in seconds
            
        Returns:
            Path to saved file
        """
        scapy = self._scapy
        
        interface = interface or self.get_default_interface()
        
        logger.info(f"Capturing to file: {filepath}")
        
        packets = scapy.sniff(
            iface=interface,
            filter=filter,
            count=count if count > 0 else 0,
            timeout=timeout
        )
        
        scapy.wrpcap(filepath, packets)
        logger.info(f"Saved {len(packets)} packets to {filepath}")
        
        return filepath


class LiveCaptureManager:
    """
    High-level manager for live capture with integrated detection.
    Suitable for integration with the AI-NIDS Flask application.
    """
    
    def __init__(self, detector=None, model_dir: str = 'models'):
        from collectors.flow_aggregator import FlowAggregator
        
        self.capture = LiveCapture()
        self.stats_callback = StatisticsCallback()
        self.queue_callback = QueueCallback()
        self.alerts: List[Dict] = []
        
        # Load or initialize detection engine if not explicitly provided
        if detector is None:
            try:
                from detection.detector import create_detection_engine
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                target_model_dir = os.path.join(project_root, model_dir)
                detector = create_detection_engine(model_dir=target_model_dir)
            except Exception as e:
                logger.warning(f"Could not auto-initialize DetectionEngine: {e}")

        self.detector = detector
        self.detection_callback = DetectionCallback(detector=self.detector)
        
        # Instantiate FlowAggregator with completed flow callback for ML detection
        self.flow_aggregator = FlowAggregator(
            idle_timeout=2.0,
            active_timeout=10.0,
            flow_callback=self._process_completed_flow
        )
        
        # Add callbacks
        self.capture.add_callback(self.stats_callback)
        self.capture.add_callback(self.queue_callback)
        self.capture.add_callback(self.flow_aggregator)
        
        if self.detector:
            self.capture.add_callback(self.detection_callback)

    def _process_completed_flow(self, flow_dict: Dict[str, Any]) -> None:
        """Callback invoked whenever FlowAggregator flushes a completed network flow."""
        if not self.detector:
            return
            
        try:
            res = self.detector.detect(flow_dict)
            if isinstance(res, list):
                res = res[0]
            
            norm_res = res.to_normalized_dict() if hasattr(res, 'to_normalized_dict') else (res.to_dict() if hasattr(res, 'to_dict') else str(res))
            flow_dict['detection'] = norm_res

            if getattr(res, 'is_attack', False):
                alert = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'flow_id': flow_dict.get('flow_id'),
                    'source_ip': flow_dict.get('source_ip'),
                    'destination_ip': flow_dict.get('destination_ip'),
                    'source_port': flow_dict.get('source_port'),
                    'destination_port': flow_dict.get('destination_port'),
                    'protocol': flow_dict.get('protocol'),
                    'attack_type': norm_res.get('attack_type', 'Unknown Attack') if isinstance(norm_res, dict) else 'Unknown Attack',
                    'confidence': norm_res.get('confidence', 0.0) if isinstance(norm_res, dict) else 0.0,
                    'severity': norm_res.get('severity', 'HIGH') if isinstance(norm_res, dict) else 'HIGH',
                    'risk_score': norm_res.get('risk_score', 0.0) if isinstance(norm_res, dict) else 0.0,
                    'model_used': norm_res.get('model_used', 'DetectionEngine') if isinstance(norm_res, dict) else 'DetectionEngine',
                    'details': norm_res
                }
                self.alerts.append(alert)
                logger.info(f"LIVE INTRUSION DETECTED: {alert['attack_type']} (confidence: {alert['confidence']}, severity: {alert['severity']})")

            # Persist live flow and optional alert to database tables
            if isinstance(norm_res, dict):
                self._save_live_flow_to_db(flow_dict, norm_res)

        except Exception as e:
            logger.error(f"Error processing completed flow through DetectionEngine: {e}")

    def _save_live_flow_to_db(self, flow_dict: Dict[str, Any], norm_res: Dict[str, Any]) -> None:
        """
        Persist completed live network flow and alert to database (NetworkFlow & Alert models)
        with strict deduplication and logical separation via batch_id='LIVE_CAPTURE'.
        """
        try:
            from app import db, create_app
            from app.models.database import NetworkFlow, Alert
            from flask import has_app_context

            def _persist():
                flow_id = str(flow_dict.get('flow_id', ''))
                src_ip = str(flow_dict.get('source_ip', flow_dict.get('src_ip', '')))
                dst_ip = str(flow_dict.get('destination_ip', flow_dict.get('dst_ip', '')))

                def _safe_int(val, default=0):
                    if val is None:
                        return default
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        return default

                def _safe_float(val, default=0.0):
                    if val is None:
                        return default
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return default

                src_port = _safe_int(flow_dict.get('source_port', flow_dict.get('src_port')), 0)
                dst_port = _safe_int(flow_dict.get('destination_port', flow_dict.get('dst_port')), 0)
                protocol = str(flow_dict.get('protocol', 'TCP'))

                # Parse timestamp
                ts = flow_dict.get('timestamp')
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except Exception:
                        ts = datetime.utcnow()
                elif not isinstance(ts, datetime):
                    ts = datetime.utcnow()

                is_attack = bool(norm_res.get('prediction', 0) == 1 or norm_res.get('is_attack', False))
                attack_type = str(norm_res.get('attack_type', 'Normal'))

                # --- 1. NetworkFlow Record Deduplication & Insertion/Update ---
                existing_flow = None
                if flow_id:
                    existing_flow = NetworkFlow.query.filter(
                        NetworkFlow.batch_id == 'LIVE_CAPTURE',
                        NetworkFlow.raw_data.like(f'%"flow_id": "{flow_id}"%')
                    ).first()

                if not existing_flow and src_ip and dst_ip:
                    existing_flow = NetworkFlow.query.filter_by(
                        batch_id='LIVE_CAPTURE',
                        source_ip=src_ip,
                        destination_ip=dst_ip,
                        source_port=src_port,
                        destination_port=dst_port,
                        protocol=protocol
                    ).order_by(NetworkFlow.timestamp.desc()).first()

                raw_json_str = json.dumps({'flow_id': flow_id, 'detection': norm_res})

                if existing_flow:
                    existing_flow.duration = _safe_float(flow_dict.get('duration'))
                    existing_flow.total_bytes = _safe_int(flow_dict.get('total_bytes'))
                    existing_flow.packets_sent = _safe_int(flow_dict.get('packets_sent'))
                    existing_flow.packets_recv = _safe_int(flow_dict.get('packets_recv'))
                    existing_flow.bytes_sent = _safe_int(flow_dict.get('bytes_sent'))
                    existing_flow.bytes_recv = _safe_int(flow_dict.get('bytes_recv'))
                    existing_flow.syn_count = _safe_int(flow_dict.get('syn_count'))
                    existing_flow.ack_count = _safe_int(flow_dict.get('ack_count'))
                    existing_flow.fin_count = _safe_int(flow_dict.get('fin_count'))
                    existing_flow.rst_count = _safe_int(flow_dict.get('rst_count'))
                    existing_flow.predicted_label = attack_type
                    existing_flow.is_anomaly = is_attack
                    existing_flow.raw_data = raw_json_str
                else:
                    new_flow = NetworkFlow(
                        timestamp=ts,
                        source_ip=src_ip,
                        destination_ip=dst_ip,
                        source_port=src_port,
                        destination_port=dst_port,
                        protocol=protocol,
                        duration=_safe_float(flow_dict.get('duration')),
                        total_bytes=_safe_int(flow_dict.get('total_bytes')),
                        packets_sent=_safe_int(flow_dict.get('packets_sent')),
                        packets_recv=_safe_int(flow_dict.get('packets_recv')),
                        bytes_sent=_safe_int(flow_dict.get('bytes_sent')),
                        bytes_recv=_safe_int(flow_dict.get('bytes_recv')),
                        syn_count=_safe_int(flow_dict.get('syn_count')),
                        ack_count=_safe_int(flow_dict.get('ack_count')),
                        fin_count=_safe_int(flow_dict.get('fin_count')),
                        rst_count=_safe_int(flow_dict.get('rst_count')),
                        label='LIVE',
                        predicted_label=attack_type,
                        is_anomaly=is_attack,
                        batch_id='LIVE_CAPTURE',
                        raw_data=raw_json_str
                    )
                    db.session.add(new_flow)

                # --- 2. Alert Record Deduplication & Insertion (if malicious) ---
                if is_attack:
                    existing_alert = None
                    if flow_id:
                        existing_alert = Alert.query.filter(
                            Alert.batch_id == 'LIVE_CAPTURE',
                            Alert.raw_data.like(f'%"flow_id": "{flow_id}"%')
                        ).first()

                    if not existing_alert:
                        severity = str(norm_res.get('severity', 'HIGH')).lower()
                        confidence = _safe_float(norm_res.get('confidence'))
                        risk_score = _safe_float(norm_res.get('risk_score'), confidence * 100.0)
                        model_used = str(norm_res.get('model_used', 'DetectionEngine'))

                        new_alert = Alert(
                            timestamp=ts,
                            source_ip=src_ip,
                            destination_ip=dst_ip,
                            source_port=src_port,
                            destination_port=dst_port,
                            protocol=protocol,
                            attack_type=attack_type,
                            severity=severity,
                            confidence=confidence,
                            risk_score=risk_score,
                            description=f"Live intrusion detected: {attack_type} from {src_ip}:{src_port} to {dst_ip}:{dst_port} (confidence: {confidence:.2f})",
                            model_used=model_used,
                            acknowledged=False,
                            resolved=False,
                            batch_id='LIVE_CAPTURE',
                            raw_data=raw_json_str
                        )
                        db.session.add(new_alert)

                db.session.commit()

            if has_app_context():
                _persist()
            else:
                app = create_app()
                with app.app_context():
                    _persist()

        except Exception as e:
            logger.error(f"Failed to persist live flow/alert to database: {e}")
            try:
                from app import db
                db.session.rollback()
            except Exception:
                pass
    
    def start_capture(
        self,
        interface: Optional[str] = None,
        filter: str = "ip or ip6",
        timeout: Optional[int] = None
    ) -> bool:
        """Start live capture with detection."""
        try:
            if not filter or str(filter).strip() == 'ip':
                filter = "ip or ip6"
            self.capture.start(
                interface=interface,
                filter=filter,
                timeout=timeout,
                async_capture=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            return False
    
    def stop_capture(self) -> None:
        """Stop live capture."""
        self.capture.stop()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current live capture & detection statistics from real backend data."""
        base_stats = self.stats_callback.get_statistics()
        flows_processed = getattr(self.flow_aggregator, 'total_flows_count', 0)
        if flows_processed == 0:
            completed_flows = self.flow_aggregator.get_completed_flows(limit=5000)
            flows_processed = len(completed_flows)
        threats_count = len(self.alerts)
        benign_count = max(0, flows_processed - threats_count)

        last_threat_str = "None"
        if self.alerts:
            last_a = self.alerts[-1]
            atk = last_a.get('attack_type', 'Unknown Attack')
            src = last_a.get('source_ip', '')
            dst = last_a.get('destination_ip', '')
            sev = str(last_a.get('severity', 'HIGH')).upper()
            if src and dst:
                last_threat_str = f"[{sev}] {atk} ({src} → {dst})"
            else:
                last_threat_str = f"[{sev}] {atk}"

        return {
            **base_stats,
            'packets_captured': base_stats.get('total_packets', 0),
            'flows_processed': flows_processed,
            'threats_detected': threats_count,
            'benign_count': benign_count,
            'last_threat': last_threat_str,
            'last_update': datetime.utcnow().isoformat()
        }
    
    def get_recent_packets(self, limit: int = 100) -> List[Dict]:
        """Get recent captured packets."""
        packets = self.queue_callback.get_all_packets()
        return [p.to_dict() for p in packets[-limit:]]
    
    def get_recent_flows(self, limit: int = 50) -> List[Dict]:
        """Get completed aggregated network flows."""
        return self.flow_aggregator.get_completed_flows(limit=limit)
    
    def get_alerts(self) -> List[Dict]:
        """Get detection alerts."""
        return self.detection_callback.alerts
    
    @property
    def is_running(self) -> bool:
        return self.capture.is_capturing


# Quick capture function
def quick_capture(
    duration: int = 10,
    interface: Optional[str] = None,
    filter: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quick packet capture with statistics.
    
    Args:
        duration: Capture duration in seconds
        interface: Network interface
        filter: BPF filter
        verbose: Print packet info
        
    Returns:
        Capture statistics
    """
    capture = LiveCapture()
    stats = StatisticsCallback()
    
    capture.add_callback(stats)
    if verbose:
        capture.add_callback(PrintCallback(verbose=True))
    
    print(f"Capturing for {duration} seconds...")
    capture.start(
        interface=interface,
        filter=filter,
        timeout=duration,
        async_capture=False
    )
    
# Factory function & Compatibility aliases
def create_live_capture() -> LiveCapture:
    """Factory function to create a LiveCapture instance."""
    return LiveCapture()


IntrusionDetector = DetectionCallback


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("AI-NIDS Live Capture Module")
    print("=" * 60)
    
    # List interfaces
    print("\nAvailable interfaces:")
    for iface in LiveCapture.list_interfaces():
        print(f"  - {iface['name']} (MAC: {iface['mac']})")
    
    print(f"\nDefault interface: {LiveCapture.get_default_interface()}")
    
    # Quick capture demo
    if len(sys.argv) > 1 and sys.argv[1] == '--capture':
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        stats = quick_capture(duration=duration, verbose=True)
        
        print("\n" + "=" * 60)
        print("Capture Statistics:")
        print("=" * 60)
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("\nUsage: python live_capture.py --capture [duration]")
        print("Example: python live_capture.py --capture 30")
