#!/usr/bin/env python3
"""
Performance Metrics Calculator for ML-NIDS
==========================================
Calculates classification and detection metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)
from typing import Dict, Any, Optional
from .logger import setup_logger

logger = setup_logger(__name__)


class MetricsCalculator:
    """Calculate and store performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.confusion_mat: Optional[np.ndarray] = None
        
    def calculate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            average: Averaging method for multiclass
            
        Returns:
            Dictionary of metrics
        """
        try:
            self.metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
                'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            }
            
            # Calculate confusion matrix
            self.confusion_mat = confusion_matrix(y_true, y_pred)
            
            # Calculate detection-specific metrics from confusion matrix
            if len(np.unique(y_true)) == 2:  # Binary classification
                tn, fp, fn, tp = self.confusion_mat.ravel()
                
                # Detection Rate (True Positive Rate / Recall)
                self.metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # False Positive Rate
                self.metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                # False Negative Rate
                self.metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
                
                # Specificity (True Negative Rate)
                self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                
            # AUC-ROC if probabilities provided
            if y_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        self.metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
                        self.metrics['average_precision'] = average_precision_score(y_true, y_proba)
                    else:
                        self.metrics['auc_roc'] = roc_auc_score(
                            y_true, y_proba, multi_class='ovr', average='weighted'
                        )
                except Exception as e:
                    logger.warning(f"Could not calculate AUC: {e}")
                    
            logger.info(f"Metrics calculated - Accuracy: {self.metrics['accuracy']:.4f}, "
                       f"F1: {self.metrics['f1_score']:.4f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
    
    def get_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        target_names: Optional[list] = None
    ) -> str:
        """Generate detailed classification report."""
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    def print_summary(self):
        """Print formatted metrics summary."""
        print("\n" + "="*50)
        print("        PERFORMANCE METRICS SUMMARY")
        print("="*50)
        
        for metric, value in self.metrics.items():
            print(f"  {metric.replace('_', ' ').title():.<30} {value:.4f}")
        
        if self.confusion_mat is not None:
            print("\nConfusion Matrix:")
            print(self.confusion_mat)
        
        print("="*50 + "\n")
