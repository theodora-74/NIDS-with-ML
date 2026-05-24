#!/usr/bin/env python3
"""Generate performance report for dissertation."""

import sys
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import config
from src.data.dataset_loader import DatasetLoader
from src.detection.hybrid_detector import HybridDetector

def generate_report():
    """Generate comprehensive performance report."""
    
    print("="*60)
    print("   ML-NIDS PERFORMANCE REPORT FOR DISSERTATION")
    print("="*60)
    
    config.load()
    
    # Load model and processor
    model_path = os.path.join(config.get('paths.models_dir'), 'hybrid_detector.joblib')
    processor_path = os.path.join(config.get('paths.models_dir'), 'feature_processor.joblib')
    
    detector = HybridDetector.load(model_path)
    feature_processor = joblib.load(processor_path)
    
    # Load test data
    data_loader = DatasetLoader()
    _, test_df = data_loader.load_nslkdd('/opt/adaptive_nids/data/datasets/NSL-KDD')
    
    if test_df is None:
        print("No test data found, using train data split")
        train_df, _ = data_loader.load_nslkdd('/opt/adaptive_nids/data/datasets/NSL-KDD')
        _, test_df = data_loader.prepare_train_test(train_df, test_size=0.2)
    
    # Process
    X_test, y_test = feature_processor.transform(test_df)
    
    # Predictions
    y_pred = detector.predict(X_test)
    y_proba = detector.predict_proba(X_test)
    
    # Classification Report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred))
    
    # Save report to file
    report_path = '/opt/adaptive_nids/output/reports/performance_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("ML-NIDS PERFORMANCE REPORT\n")
        f.write("="*60 + "\n\n")
        f.write("Dataset: NSL-KDD\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n\nMODEL ARCHITECTURE:\n")
        f.write("- Ensemble: XGBoost (30%), LightGBM (30%), Random Forest (25%), Gradient Boosting (15%)\n")
        f.write("- Anomaly Detection: Isolation Forest, One-Class SVM, Autoencoder (PyTorch)\n")
    
    print(f"\nReport saved to: {report_path}")
    
    # Generate Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - ML-NIDS')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = '/opt/adaptive_nids/output/reports/confusion_matrix.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Confusion matrix saved to: {plot_path}")
    
    print("\n" + "="*60)
    print("   REPORT GENERATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    generate_report()
