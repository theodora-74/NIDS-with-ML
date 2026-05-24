#!/usr/bin/env python3
"""
Real-Time Network Intrusion Detection
======================================
Captures live network traffic and detects attacks in real-time.
"""

import sys
import os
import time
import signal
import argparse
import numpy as np
import joblib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import config
from src.detection.hybrid_detector import HybridDetector
from src.realtime.packet_capture import PacketCapture

# Colors
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_banner():
    """Print animated banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     █████╗ ██████╗  █████╗ ██████╗ ████████╗██╗██╗   ██╗███████╗ ║
║    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██║   ██║██╔════╝ ║
║    ███████║██║  ██║███████║██████╔╝   ██║   ██║██║   ██║█████╗   ║
║    ██╔══██║██║  ██║██╔══██║██╔═══╝    ██║   ██║╚██╗ ██╔╝██╔══╝   ║
║    ██║  ██║██████╔╝██║  ██║██║        ██║   ██║ ╚████╔╝ ███████╗ ║
║    ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝  ╚═══╝  ╚══════╝ ║
║                                                                  ║
║            ███╗   ███╗██╗      ███╗   ██╗██╗██████╗ ███████╗     ║
║            ████╗ ████║██║      ████╗  ██║██║██╔══██╗██╔════╝     ║
║            ██╔████╔██║██║█████╗██╔██╗ ██║██║██║  ██║███████╗     ║
║            ██║╚██╔╝██║██║╚════╝██║╚██╗██║██║██║  ██║╚════██║     ║
║            ██║ ╚═╝ ██║███████╗ ██║ ╚████║██║██████╔╝███████║     ║
║            ╚═╝     ╚═╝╚══════╝ ╚═╝  ╚═══╝╚═╝╚═════╝ ╚══════╝     ║
║                                                                  ║
║        Machine Learning Network Intrusion Detection System       ║
║                                                                  ║
║                    BSc Dissertation Project                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.END}
"""
    # Animate banner line by line
    for line in banner.split('\n'):
        print(line)
        time.sleep(0.05)


def print_loading_animation():
    """Print loading animation."""
    stages = [
        "Initializing system",
        "Loading ML models",
        "Preparing detection engine",
        "Calibrating sensors",
        "Starting network capture"
    ]
    
    for stage in stages:
        sys.stdout.write(f"\r  {Colors.YELLOW}⟳{Colors.END} {stage}...")
        sys.stdout.flush()
        time.sleep(0.5)
        sys.stdout.write(f"\r  {Colors.GREEN}✓{Colors.END} {stage}   \n")
        sys.stdout.flush()


class RealTimeDetector:
    """Real-time network intrusion detection system."""
    
    def __init__(self, interface: str = "eth0"):
        """Initialize real-time detector."""
        print_banner()
        time.sleep(0.5)
        
        print(f"\n{Colors.BOLD}INITIALIZING SYSTEM{Colors.END}")
        print("="*60)
        
        # Load configuration
        config.load()
        
        # Loading animation
        sys.stdout.write(f"  {Colors.YELLOW}⟳{Colors.END} Loading ML Model...")
        sys.stdout.flush()
        time.sleep(0.3)
        model_path = '/opt/adaptive_nids/models/trained/hybrid_detector.joblib'
        self.detector = HybridDetector.load(model_path)
        sys.stdout.write(f"\r  {Colors.GREEN}✓{Colors.END} Loading ML Model...Done!   \n")
        
        sys.stdout.write(f"  {Colors.YELLOW}⟳{Colors.END} Loading Feature Processor...")
        sys.stdout.flush()
        time.sleep(0.3)
        processor_path = '/opt/adaptive_nids/models/trained/feature_processor.joblib'
        self.feature_processor = joblib.load(processor_path)
        sys.stdout.write(f"\r  {Colors.GREEN}✓{Colors.END} Loading Feature Processor...Done!   \n")
        
        sys.stdout.write(f"  {Colors.YELLOW}⟳{Colors.END} Initializing Packet Capture...")
        sys.stdout.flush()
        time.sleep(0.3)
        self.capture = PacketCapture(interface=interface, flow_timeout=30)
        sys.stdout.write(f"\r  {Colors.GREEN}✓{Colors.END} Initializing Packet Capture on {interface}...Done!   \n")
        
        # Statistics
        self.stats = {
            'total_flows': 0,
            'normal': 0,
            'attacks': 0,
            'anomalies': 0
        }
        
        self.running = False
        
        print("="*60)
        print(f"  {Colors.GREEN}{Colors.BOLD}✓ SYSTEM READY!{Colors.END}\n")
        time.sleep(0.5)
    
    def _print_alert(self, flow, prediction, confidence, anomaly_score, alert_level):
        """Print detection alert with visual effects."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if alert_level == 2:
            # Attack alert with blinking effect
            print(f"\n{Colors.RED}{Colors.BOLD}{'!'*60}{Colors.END}")
            print(f"{Colors.RED}{Colors.BOLD}⚠⚠⚠  ATTACK DETECTED  ⚠⚠⚠{Colors.END}")
            print(f"{Colors.RED}{Colors.BOLD}{'!'*60}{Colors.END}")
            status_color = Colors.RED
            self.stats['attacks'] += 1
        elif alert_level == 1:
            print(f"\n{Colors.YELLOW}{'~'*60}{Colors.END}")
            print(f"{Colors.YELLOW}{Colors.BOLD}⚡ ANOMALY DETECTED ⚡{Colors.END}")
            print(f"{Colors.YELLOW}{'~'*60}{Colors.END}")
            status_color = Colors.YELLOW
            self.stats['anomalies'] += 1
        else:
            status_color = Colors.GREEN
            self.stats['normal'] += 1
        
        self.stats['total_flows'] += 1
        
        # Only print details for alerts
        if alert_level > 0:
            print(f"\n  {Colors.BOLD}Timestamp:{Colors.END}   {timestamp}")
            print(f"  {Colors.BOLD}Source:{Colors.END}      {status_color}{flow.src_ip}:{flow.src_port}{Colors.END}")
            print(f"  {Colors.BOLD}Destination:{Colors.END} {status_color}{flow.dst_ip}:{flow.dst_port}{Colors.END}")
            print(f"  {Colors.BOLD}Protocol:{Colors.END}    {flow.protocol.upper()}")
            print(f"  {Colors.BOLD}Service:{Colors.END}     {flow.service}")
            print(f"  {Colors.BOLD}Packets:{Colors.END}     {flow.packet_count}")
            print(f"  {Colors.BOLD}Bytes:{Colors.END}       {flow.total_bytes}")
            print(f"  {Colors.BOLD}Duration:{Colors.END}    {flow.duration:.2f}s")
            print(f"  {Colors.BOLD}Prediction:{Colors.END}  {status_color}{prediction}{Colors.END}")
            print(f"  {Colors.BOLD}Confidence:{Colors.END}  {confidence:.2%}")
            print(f"  {Colors.BOLD}Anomaly:{Colors.END}     {anomaly_score:.4f}")
            
            if alert_level == 2:
                print(f"\n  {Colors.RED}{Colors.BOLD}⛔ ACTION REQUIRED: Block/Investigate this connection!{Colors.END}")
                print(f"{Colors.RED}{Colors.BOLD}{'!'*60}{Colors.END}\n")
            else:
                print(f"{Colors.YELLOW}{'~'*60}{Colors.END}\n")
    
    def _print_stats(self):
        """Print current statistics with visual formatting."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}╔{'═'*58}╗{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}║{'LIVE STATISTICS':^58}║{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}╠{'═'*58}╣{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}║{Colors.END}  Total Flows Analyzed: {self.stats['total_flows']:<33}{Colors.CYAN}{Colors.BOLD}║{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}║{Colors.END}  {Colors.GREEN}Normal Traffic:{Colors.END}       {self.stats['normal']:<33}{Colors.CYAN}{Colors.BOLD}║{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}║{Colors.END}  {Colors.RED}Attacks Detected:{Colors.END}     {self.stats['attacks']:<33}{Colors.CYAN}{Colors.BOLD}║{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}║{Colors.END}  {Colors.YELLOW}Anomalies Found:{Colors.END}      {self.stats['anomalies']:<33}{Colors.CYAN}{Colors.BOLD}║{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}║{Colors.END}  Total Packets:        {self.capture.total_packets:<33}{Colors.CYAN}{Colors.BOLD}║{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}║{Colors.END}  Total Bytes:          {self.capture.total_bytes:<33}{Colors.CYAN}{Colors.BOLD}║{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}╚{'═'*58}╝{Colors.END}\n")
    
    def _print_monitoring_status(self):
        """Print monitoring status bar."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        active_flows = len(self.capture.get_active_flows())
        
        status = f"[{timestamp}] Monitoring... | Active Flows: {active_flows} | "
        status += f"{Colors.GREEN}Normal: {self.stats['normal']}{Colors.END} | "
        status += f"{Colors.RED}Attacks: {self.stats['attacks']}{Colors.END} | "
        status += f"{Colors.YELLOW}Anomalies: {self.stats['anomalies']}{Colors.END}"
        
        sys.stdout.write(f"\r{status}    ")
        sys.stdout.flush()
    
    def start(self, verbose: bool = False):
        """Start real-time detection."""
        self.running = True
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print(f"\n\n{Colors.YELLOW}{Colors.BOLD}Stopping detection...{Colors.END}")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start packet capture
        self.capture.start()
        
        print(f"{Colors.GREEN}{Colors.BOLD}╔{'═'*58}╗{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}║{'REAL-TIME DETECTION ACTIVE':^58}║{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}╚{'═'*58}╝{Colors.END}")
        print(f"\n  Interface: {Colors.CYAN}{self.capture.interface}{Colors.END}")
        print(f"  Press {Colors.YELLOW}Ctrl+C{Colors.END} to stop\n")
        print("-"*60)
        
        last_stats_time = time.time()
        last_status_time = time.time()
        
        try:
            while self.running:
                # Get completed flows
                flows = self.capture.get_completed_flows()
                
                if flows:
                    # Extract features
                    features = self.capture.get_flow_features(flows)
                    
                    if len(features) > 0:
                        try:
                            features_scaled = self.feature_processor.scaler.transform(features)
                        except Exception:
                            features_scaled = features
                        
                        # Run detection
                        predictions = self.detector.predict(features_scaled)
                        results = self.detector.detect_with_anomaly(features_scaled)
                        
                        # Process each flow
                        for i, flow in enumerate(flows):
                            prediction = predictions[i]
                            confidence = results['ensemble_confidence'][i]
                            anomaly_score = results['anomaly_score'][i]
                            alert_level = results['final_alert'][i]
                            
                            self._print_alert(flow, prediction, confidence, anomaly_score, alert_level)
                            
                            if verbose and alert_level == 0:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                print(f"[{timestamp}] {Colors.GREEN}✓{Colors.END} {flow.src_ip}:{flow.src_port} → {flow.dst_ip}:{flow.dst_port} ({flow.protocol})")
                
                # Update status bar every 2 seconds
                if time.time() - last_status_time > 2:
                    self._print_monitoring_status()
                    last_status_time = time.time()
                
                # Print stats every 60 seconds
                if time.time() - last_stats_time > 60:
                    print()  # Clear status line
                    self._print_stats()
                    last_stats_time = time.time()
                
                time.sleep(0.5)
                
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        
        finally:
            self.capture.stop()
            print()  # Clear status line
            self._print_stats()
            
            print(f"\n{Colors.CYAN}{Colors.BOLD}╔{'═'*58}╗{Colors.END}")
            print(f"{Colors.CYAN}{Colors.BOLD}║{'DETECTION STOPPED':^58}║{Colors.END}")
            print(f"{Colors.CYAN}{Colors.BOLD}║{'Thank you for using Adaptive ML-NIDS':^58}║{Colors.END}")
            print(f"{Colors.CYAN}{Colors.BOLD}╚{'═'*58}╝{Colors.END}\n")


def main():
    parser = argparse.ArgumentParser(description='Real-Time Network Intrusion Detection')
    parser.add_argument('-i', '--interface', default='eth0', help='Network interface to capture')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show all traffic (including normal)')
    
    args = parser.parse_args()
    
    # Check for root privileges
    if os.geteuid() != 0:
        print_banner()
        print(f"\n{Colors.RED}{Colors.BOLD}ERROR: Root privileges required for packet capture.{Colors.END}")
        print(f"\nRun with: {Colors.CYAN}sudo python3 realtime_detection.py -i {args.interface}{Colors.END}\n")
        sys.exit(1)
    
    detector = RealTimeDetector(interface=args.interface)
    detector.start(verbose=args.verbose)


if __name__ == '__main__':
    main()
