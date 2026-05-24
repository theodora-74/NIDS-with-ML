#!/usr/bin/env python3
"""
LIVE DEMO for Dissertation Presentation
========================================
Shows real-time attack detection with visual output.
"""

import sys
import os
import time
import numpy as np
import joblib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import config
from src.data.dataset_loader import DatasetLoader
from src.detection.hybrid_detector import HybridDetector

# Colors for terminal
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
    """Print welcome banner."""
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
    print(banner)

def print_system_info():
    """Print system information."""
    print(f"\n{Colors.BOLD}SYSTEM INFORMATION{Colors.END}")
    print("="*60)
    print(f"  {Colors.GREEN}►{Colors.END} Ensemble Models: XGBoost, LightGBM, Random Forest, Gradient Boosting")
    print(f"  {Colors.GREEN}►{Colors.END} Anomaly Detection: Isolation Forest, One-Class SVM, Autoencoder")
    print(f"  {Colors.GREEN}►{Colors.END} Dataset: NSL-KDD (125,973 training / 22,544 test samples)")
    print(f"  {Colors.GREEN}►{Colors.END} Features: 41 network traffic attributes")
    print("="*60)

def simulate_realtime_detection(detector, feature_processor, test_samples, labels):
    """Simulate real-time detection with visual output."""
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}STARTING REAL-TIME DETECTION SIMULATION{Colors.END}")
    print("="*60)
    print(f"  Monitoring network traffic...")
    print(f"  Press Ctrl+C to stop\n")
    
    attack_types = {
        'normal': f'{Colors.GREEN}[NORMAL]{Colors.END}',
        'neptune': f'{Colors.RED}[DoS: NEPTUNE]{Colors.END}',
        'satan': f'{Colors.RED}[PROBE: SATAN]{Colors.END}',
        'ipsweep': f'{Colors.YELLOW}[PROBE: IPSWEEP]{Colors.END}',
        'portsweep': f'{Colors.YELLOW}[PROBE: PORTSWEEP]{Colors.END}',
        'smurf': f'{Colors.RED}[DoS: SMURF]{Colors.END}',
        'nmap': f'{Colors.YELLOW}[PROBE: NMAP]{Colors.END}',
        'back': f'{Colors.RED}[DoS: BACK]{Colors.END}',
        'teardrop': f'{Colors.RED}[DoS: TEARDROP]{Colors.END}',
        'warezclient': f'{Colors.MAGENTA}[R2L: WAREZCLIENT]{Colors.END}',
        'pod': f'{Colors.RED}[DoS: POD]{Colors.END}',
        'guess_passwd': f'{Colors.MAGENTA}[R2L: GUESS_PASSWD]{Colors.END}',
        'buffer_overflow': f'{Colors.RED}[U2R: BUFFER_OVERFLOW]{Colors.END}',
        'warezmaster': f'{Colors.MAGENTA}[R2L: WAREZMASTER]{Colors.END}',
        'land': f'{Colors.RED}[DoS: LAND]{Colors.END}',
        'imap': f'{Colors.MAGENTA}[R2L: IMAP]{Colors.END}',
        'rootkit': f'{Colors.RED}[U2R: ROOTKIT]{Colors.END}',
        'multihop': f'{Colors.MAGENTA}[R2L: MULTIHOP]{Colors.END}',
        'phf': f'{Colors.MAGENTA}[R2L: PHF]{Colors.END}',
        'ftp_write': f'{Colors.MAGENTA}[R2L: FTP_WRITE]{Colors.END}',
        'spy': f'{Colors.MAGENTA}[R2L: SPY]{Colors.END}',
    }
    
    stats = {'normal': 0, 'attack': 0, 'anomaly': 0}
    
    try:
        # Process in small batches to simulate real-time
        batch_size = 5
        total_processed = 0
        
        for i in range(0, min(len(test_samples), 200), batch_size):
            batch_X = test_samples[i:i+batch_size]
            batch_y = labels[i:i+batch_size]
            
            # Get predictions
            predictions = detector.predict(batch_X)
            results = detector.detect_with_anomaly(batch_X)
            
            for j in range(len(batch_X)):
                total_processed += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                true_label = str(batch_y[j]).lower() if batch_y is not None else "unknown"
                pred_label = str(predictions[j]).lower()
                alert_level = results['final_alert'][j]
                confidence = results['ensemble_confidence'][j]
                anomaly_score = results['anomaly_score'][j]
                
                # Get attack type display
                attack_display = attack_types.get(true_label, f'{Colors.YELLOW}[{true_label.upper()}]{Colors.END}')
                
                # Determine status
                if alert_level == 2:
                    status = f"{Colors.RED}{Colors.BOLD}HIGH ALERT{Colors.END}"
                    stats['attack'] += 1
                elif alert_level == 1:
                    status = f"{Colors.YELLOW}ANOMALY{Colors.END}"
                    stats['anomaly'] += 1
                else:
                    status = f"{Colors.GREEN} NORMAL{Colors.END}"
                    stats['normal'] += 1
                
                # Print detection result
                src_ip = f"192.168.1.{np.random.randint(1, 255)}"
                dst_ip = f"10.0.0.{np.random.randint(1, 255)}"
                src_port = np.random.randint(1024, 65535)
                dst_port = np.random.choice([22, 23, 80, 443, 3389, 445, 21])
                
                print(f"[{timestamp}] {src_ip}:{src_port} → {dst_ip}:{dst_port}")
                print(f"  {status} | True: {attack_display} | Confidence: {confidence:.2%} | Anomaly: {anomaly_score:.2f}")
                print()
                
                time.sleep(0.3)  # Simulate real-time delay
        
        # Print final statistics
        print("\n" + "="*60)
        print(f"{Colors.BOLD}DETECTION SESSION SUMMARY{Colors.END}")
        print("="*60)
        print(f"  Total Packets Analyzed: {total_processed}")
        print(f"  {Colors.GREEN}Normal Traffic:{Colors.END}    {stats['normal']}")
        print(f"  {Colors.RED}Attacks Detected:{Colors.END}  {stats['attack']}")
        print(f"  {Colors.YELLOW}Anomalies Found:{Colors.END}   {stats['anomaly']}")
        print("="*60)
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Detection stopped by user.{Colors.END}")
        print(f"  Processed: {total_processed} packets")

def show_model_performance():
    """Display model performance metrics."""
    print(f"\n{Colors.BOLD}MODEL PERFORMANCE ON NSL-KDD TEST SET{Colors.END}")
    print("="*60)
    
    metrics = [
        ("Overall Accuracy", "72.23%"),
        ("Weighted Precision", "57.16%"),
        ("Weighted Recall", "72.23%"),
        ("Weighted F1-Score", "61.92%"),
    ]
    
    for name, value in metrics:
        print(f"  {name:.<40} {Colors.CYAN}{value}{Colors.END}")
    
    print("\n" + "-"*60)
    print(f"{Colors.BOLD}TOP PERFORMING ATTACK DETECTION:{Colors.END}")
    print("-"*60)
    
    top_attacks = [
        ("Neptune (DoS)", "97%", "100%", "98%"),
        ("Smurf (DoS)", "100%", "99%", "100%"),
        ("Nmap (Probe)", "100%", "99%", "99%"),
        ("Portsweep (Probe)", "80%", "92%", "86%"),
        ("Satan (Probe)", "65%", "100%", "79%"),
    ]
    
    print(f"  {'Attack Type':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("  " + "-"*56)
    for attack, prec, rec, f1 in top_attacks:
        print(f"  {attack:<20} {Colors.GREEN}{prec:<12}{Colors.END} {Colors.GREEN}{rec:<12}{Colors.END} {Colors.GREEN}{f1:<12}{Colors.END}")
    
    print("="*60)

def main():
    """Main demo function."""
    print_banner()
    time.sleep(1)
    
    print(f"\n{Colors.YELLOW}Loading ML-NIDS System...{Colors.END}")
    
    # Load configuration
    config.load()
    
    # Load model and processor
    model_path = '/opt/adaptive_nids/models/trained/hybrid_detector.joblib'
    processor_path = '/opt/adaptive_nids/models/trained/feature_processor.joblib'
    
    print(f"  {Colors.GREEN}✓{Colors.END} Loading Hybrid Detector...")
    detector = HybridDetector.load(model_path)
    
    print(f"  {Colors.GREEN}✓{Colors.END} Loading Feature Processor...")
    feature_processor = joblib.load(processor_path)
    
    print(f"  {Colors.GREEN}✓{Colors.END} Loading Test Dataset...")
    data_loader = DatasetLoader()
    _, test_df = data_loader.load_nslkdd('/opt/adaptive_nids/data/datasets/NSL-KDD')
    
    X_test, y_test = feature_processor.transform(test_df)
    
    print(f"  {Colors.GREEN}✓{Colors.END} System Ready!")
    
    print_system_info()
    
    while True:
        print(f"\n{Colors.BOLD}DEMO MENU{Colors.END}")
        print("="*40)
        print("  1. Real-Time Detection Simulation")
        print("  2. Show Model Performance")
        print("  3. Exit")
        print("="*40)
        
        choice = input(f"\n{Colors.CYAN}Select option (1-3): {Colors.END}")
        
        if choice == '1':
            simulate_realtime_detection(detector, feature_processor, X_test, y_test)
        elif choice == '2':
            show_model_performance()
        elif choice == '3':
            print(f"\n{Colors.GREEN}Thank you for using ML-NIDS!{Colors.END}\n")
            break
        else:
            print(f"{Colors.RED}Invalid option. Please try again.{Colors.END}")

if __name__ == '__main__':
    main()
