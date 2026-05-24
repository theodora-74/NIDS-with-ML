#!/usr/bin/env python3
"""
Real-Time Packet Capture and Flow Analysis
==========================================
Captures live network traffic and extracts features for ML detection.
"""

import time
import threading
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy not available. Install with: pip install scapy")

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class NetworkFlow:
    """Represents a network flow (connection)."""
    
    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: str):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        
        # Timing
        self.start_time = time.time()
        self.last_packet_time = time.time()
        
        # Counters
        self.packet_count = 0
        self.total_bytes = 0
        self.fwd_packets = 0
        self.bwd_packets = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        
        # Packet sizes
        self.packet_sizes: List[int] = []
        self.fwd_packet_sizes: List[int] = []
        self.bwd_packet_sizes: List[int] = []
        
        # Inter-arrival times
        self.inter_arrival_times: List[float] = []
        
        # TCP Flags
        self.syn_count = 0
        self.ack_count = 0
        self.fin_count = 0
        self.rst_count = 0
        self.psh_count = 0
        self.urg_count = 0
        
        # Service detection
        self.service = self._detect_service(dst_port)
        
        # Payload info
        self.has_payload = False
        self.payload_bytes = 0
    
    def _detect_service(self, port: int) -> str:
        """Detect service based on port number."""
        services = {
            20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet',
            25: 'smtp', 53: 'domain', 80: 'http', 110: 'pop3',
            143: 'imap', 443: 'https', 445: 'smb', 3306: 'mysql',
            3389: 'rdp', 5432: 'postgresql', 6379: 'redis',
            8080: 'http_proxy', 27017: 'mongodb'
        }
        return services.get(port, 'other')
    
    def add_packet(self, packet_size: int, is_forward: bool, tcp_flags: Dict[str, bool] = None):
        """Add a packet to this flow."""
        current_time = time.time()
        
        # Update timing
        if self.packet_count > 0:
            iat = current_time - self.last_packet_time
            self.inter_arrival_times.append(iat)
        self.last_packet_time = current_time
        
        # Update counters
        self.packet_count += 1
        self.total_bytes += packet_size
        self.packet_sizes.append(packet_size)
        
        if is_forward:
            self.fwd_packets += 1
            self.fwd_bytes += packet_size
            self.fwd_packet_sizes.append(packet_size)
        else:
            self.bwd_packets += 1
            self.bwd_bytes += packet_size
            self.bwd_packet_sizes.append(packet_size)
        
        # Update TCP flags
        if tcp_flags:
            if tcp_flags.get('SYN'): self.syn_count += 1
            if tcp_flags.get('ACK'): self.ack_count += 1
            if tcp_flags.get('FIN'): self.fin_count += 1
            if tcp_flags.get('RST'): self.rst_count += 1
            if tcp_flags.get('PSH'): self.psh_count += 1
            if tcp_flags.get('URG'): self.urg_count += 1
    
    @property
    def duration(self) -> float:
        """Get flow duration in seconds."""
        return self.last_packet_time - self.start_time
    
    def extract_features(self) -> Dict[str, float]:
        """Extract features for ML model (NSL-KDD style)."""
        duration = max(self.duration, 0.001)  # Avoid division by zero
        
        features = {
            # Basic features
            'duration': duration,
            'protocol_type': {'tcp': 0, 'udp': 1, 'icmp': 2}.get(self.protocol.lower(), 0),
            'service': hash(self.service) % 100,  # Simple encoding
            'flag': self._get_connection_flag(),
            'src_bytes': self.fwd_bytes,
            'dst_bytes': self.bwd_bytes,
            
            # Content features
            'land': 1 if (self.src_ip == self.dst_ip and self.src_port == self.dst_port) else 0,
            'wrong_fragment': 0,
            'urgent': self.urg_count,
            
            # Traffic features
            'hot': 0,
            'num_failed_logins': 0,
            'logged_in': 1 if self.ack_count > 0 else 0,
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'num_outbound_cmds': 0,
            'is_host_login': 0,
            'is_guest_login': 0,
            
            # Time-based features (approximated)
            'count': self.packet_count,
            'srv_count': self.packet_count,
            'serror_rate': self.rst_count / max(self.packet_count, 1),
            'srv_serror_rate': self.rst_count / max(self.packet_count, 1),
            'rerror_rate': 0,
            'srv_rerror_rate': 0,
            'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0,
            'srv_diff_host_rate': 0.0,
            
            # Host-based features (approximated)
            'dst_host_count': 1,
            'dst_host_srv_count': 1,
            'dst_host_same_srv_rate': 1.0,
            'dst_host_diff_srv_rate': 0.0,
            'dst_host_same_src_port_rate': 1.0,
            'dst_host_srv_diff_host_rate': 0.0,
            'dst_host_serror_rate': self.rst_count / max(self.packet_count, 1),
            'dst_host_srv_serror_rate': self.rst_count / max(self.packet_count, 1),
            'dst_host_rerror_rate': 0,
            'dst_host_srv_rerror_rate': 0,
        }
        
        return features
    
    def _get_connection_flag(self) -> int:
        """Determine connection flag based on TCP flags."""
        if self.rst_count > 0:
            return 4  # REJ/RSTO
        elif self.syn_count > 0 and self.fin_count > 0:
            return 0  # SF (normal)
        elif self.syn_count > 0 and self.ack_count == 0:
            return 1  # S0
        elif self.syn_count > 0:
            return 2  # S1
        else:
            return 0  # SF


class PacketCapture:
    """
    Real-time packet capture and flow aggregation.
    
    Captures network packets, aggregates them into flows,
    and extracts features for ML-based detection.
    """
    
    def __init__(self, interface: str = "eth0", flow_timeout: int = 60):
        """
        Initialize packet capture.
        
        Args:
            interface: Network interface to capture on
            flow_timeout: Seconds before a flow is considered complete
        """
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for packet capture")
        
        self.interface = interface
        self.flow_timeout = flow_timeout
        
        # Flow storage: key = (src_ip, dst_ip, src_port, dst_port, protocol)
        self.flows: Dict[Tuple, NetworkFlow] = {}
        self.flows_lock = threading.Lock()
        
        # Completed flows queue
        self.completed_flows: List[NetworkFlow] = []
        self.completed_lock = threading.Lock()
        
        # Statistics
        self.total_packets = 0
        self.total_bytes = 0
        
        # Control
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.cleanup_thread: Optional[threading.Thread] = None
        
        logger.info(f"PacketCapture initialized on interface: {interface}")
    
    def _get_flow_key(self, packet) -> Optional[Tuple]:
        """Extract flow key from packet."""
        try:
            if IP not in packet:
                return None
            
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = "other"
            src_port = 0
            dst_port = 0
            
            if TCP in packet:
                protocol = "tcp"
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif UDP in packet:
                protocol = "udp"
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
            elif ICMP in packet:
                protocol = "icmp"
            
            return (src_ip, dst_ip, src_port, dst_port, protocol)
        except Exception:
            return None
    
    def _get_tcp_flags(self, packet) -> Dict[str, bool]:
        """Extract TCP flags from packet."""
        flags = {'SYN': False, 'ACK': False, 'FIN': False, 
                 'RST': False, 'PSH': False, 'URG': False}
        
        if TCP in packet:
            tcp_flags = packet[TCP].flags
            flags['SYN'] = bool(tcp_flags & 0x02)
            flags['ACK'] = bool(tcp_flags & 0x10)
            flags['FIN'] = bool(tcp_flags & 0x01)
            flags['RST'] = bool(tcp_flags & 0x04)
            flags['PSH'] = bool(tcp_flags & 0x08)
            flags['URG'] = bool(tcp_flags & 0x20)
        
        return flags
    
    def _packet_callback(self, packet):
        """Process captured packet."""
        try:
            flow_key = self._get_flow_key(packet)
            if flow_key is None:
                return
            
            packet_size = len(packet)
            self.total_packets += 1
            self.total_bytes += packet_size
            
            # Get TCP flags
            tcp_flags = self._get_tcp_flags(packet)
            
            with self.flows_lock:
                # Check if flow exists (in either direction)
                reverse_key = (flow_key[1], flow_key[0], flow_key[3], flow_key[2], flow_key[4])
                
                if flow_key in self.flows:
                    self.flows[flow_key].add_packet(packet_size, True, tcp_flags)
                elif reverse_key in self.flows:
                    self.flows[reverse_key].add_packet(packet_size, False, tcp_flags)
                else:
                    # Create new flow
                    flow = NetworkFlow(
                        src_ip=flow_key[0],
                        dst_ip=flow_key[1],
                        src_port=flow_key[2],
                        dst_port=flow_key[3],
                        protocol=flow_key[4]
                    )
                    flow.add_packet(packet_size, True, tcp_flags)
                    self.flows[flow_key] = flow
                    
        except Exception as e:
            logger.debug(f"Error processing packet: {e}")
    
    def _cleanup_flows(self):
        """Periodically clean up expired flows."""
        while self.running:
            time.sleep(5)  # Check every 5 seconds
            
            current_time = time.time()
            expired_keys = []
            
            with self.flows_lock:
                for key, flow in self.flows.items():
                    if current_time - flow.last_packet_time > self.flow_timeout:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    flow = self.flows.pop(key)
                    with self.completed_lock:
                        self.completed_flows.append(flow)
    
    def start(self):
        """Start packet capture."""
        if self.running:
            logger.warning("Capture already running")
            return
        
        self.running = True
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_flows, daemon=True)
        self.cleanup_thread.start()
        
        # Start capture thread
        def capture():
            try:
                logger.info(f"Starting capture on {self.interface}")
                sniff(
                    iface=self.interface,
                    prn=self._packet_callback,
                    store=False,
                    stop_filter=lambda x: not self.running
                )
            except Exception as e:
                logger.error(f"Capture error: {e}")
        
        self.capture_thread = threading.Thread(target=capture, daemon=True)
        self.capture_thread.start()
        
        logger.info("Packet capture started")
    
    def stop(self):
        """Stop packet capture."""
        self.running = False
        logger.info("Packet capture stopped")
    
    def get_completed_flows(self) -> List[NetworkFlow]:
        """Get and clear completed flows."""
        with self.completed_lock:
            flows = self.completed_flows.copy()
            self.completed_flows.clear()
        return flows
    
    def get_active_flows(self) -> List[NetworkFlow]:
        """Get currently active flows."""
        with self.flows_lock:
            return list(self.flows.values())
    
    def get_flow_features(self, flows: List[NetworkFlow]) -> np.ndarray:
        """Extract features from flows for ML prediction."""
        if not flows:
            return np.array([])
        
        feature_list = []
        for flow in flows:
            features = flow.extract_features()
            # Convert to list in consistent order
            feature_values = [
                features['duration'],
                features['protocol_type'],
                features['service'],
                features['flag'],
                features['src_bytes'],
                features['dst_bytes'],
                features['land'],
                features['wrong_fragment'],
                features['urgent'],
                features['hot'],
                features['num_failed_logins'],
                features['logged_in'],
                features['num_compromised'],
                features['root_shell'],
                features['su_attempted'],
                features['num_root'],
                features['num_file_creations'],
                features['num_shells'],
                features['num_access_files'],
                features['num_outbound_cmds'],
                features['is_host_login'],
                features['is_guest_login'],
                features['count'],
                features['srv_count'],
                features['serror_rate'],
                features['srv_serror_rate'],
                features['rerror_rate'],
                features['srv_rerror_rate'],
                features['same_srv_rate'],
                features['diff_srv_rate'],
                features['srv_diff_host_rate'],
                features['dst_host_count'],
                features['dst_host_srv_count'],
                features['dst_host_same_srv_rate'],
                features['dst_host_diff_srv_rate'],
                features['dst_host_same_src_port_rate'],
                features['dst_host_srv_diff_host_rate'],
                features['dst_host_serror_rate'],
                features['dst_host_srv_serror_rate'],
                features['dst_host_rerror_rate'],
                features['dst_host_srv_rerror_rate'],
            ]
            feature_list.append(feature_values)
        
        return np.array(feature_list, dtype=np.float32)
