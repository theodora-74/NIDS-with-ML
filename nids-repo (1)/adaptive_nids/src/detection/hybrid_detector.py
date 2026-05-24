#!/usr/bin/env python3
"""
Hybrid Detection Engine
=======================
Combines supervised ensemble classifier with unsupervised
anomaly detection for comprehensive threat detection.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import joblib

from .base_detector import BaseDetector
from .ensemble_classifier import EnsembleClassifier
from .anomaly_detector import AnomalyDetector
from ..utils.logger import setup_logger
from ..utils.metrics import MetricsCalculator

logger = setup_logger(__name__)


class HybridDetector(BaseDetector):
    """
    Hybrid Detection System combining:
    
    1. SUPERVISED: Ensemble Classifier (XGBoost + LightGBM + RF + GB)
       - Detects KNOWN attack patterns
       - High accuracy on trained attack types
       
    2. UNSUPERVISED: Anomaly Detector (Isolation Forest + OCSVM + Autoencoder)
       - Detects UNKNOWN/ZERO-DAY attacks
       - Finds patterns that deviate from normal
    
    Decision Logic:
    ---------------
    1. First check: Is this a KNOWN attack? (Ensemble)
    2. If classified as normal, double-check: Is this an ANOMALY? (Anomaly Detector)
    3. High confidence attack → Alert
    4. Normal but anomalous → Flag for review (potential zero-day)
    """
    
    def __init__(self):
        """Initialize hybrid detector."""
        super().__init__("HybridDetector")
        
        self.ensemble = EnsembleClassifier()
        self.anomaly_detector = AnomalyDetector()
        self.metrics = MetricsCalculator()
        
        # Thresholds for decision making
        self.ensemble_confidence_threshold = 0.7
        self.anomaly_score_threshold = 0.5
        
        logger.info("HybridDetector initialized")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train both detection systems.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("="*60)
        logger.info("      TRAINING HYBRID DETECTION SYSTEM")
        logger.info("="*60)
        
        # Train supervised ensemble
        logger.info("\n[Phase 1/2] Training Supervised Ensemble...")
        self.ensemble.train(X, y)
        
        # Train unsupervised anomaly detector (on normal data only)
        logger.info("\n[Phase 2/2] Training Anomaly Detector...")
        self.anomaly_detector.train(X, y)
        
        self.is_trained = True
        
        logger.info("\n" + "="*60)
        logger.info("      HYBRID SYSTEM TRAINING COMPLETE")
        logger.info("="*60)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using hybrid approach.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise RuntimeError("HybridDetector not trained")
        
        # Get ensemble predictions
        ensemble_pred = self.ensemble.predict(X)
        
        return ensemble_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability array
        """
        return self.ensemble.predict_proba(X)
    
    def detect_with_anomaly(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full detection with both systems.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary containing:
            - ensemble_prediction: Classification labels
            - ensemble_confidence: Confidence scores
            - anomaly_score: Anomaly scores
            - is_anomaly: Boolean anomaly flags
            - final_alert: Combined alert status
        """
        if not self.is_trained:
            raise RuntimeError("HybridDetector not trained")
        
        results = {}
        
        # Ensemble classification
        ensemble_proba = self.ensemble.predict_proba(X)
        results['ensemble_prediction'] = self.ensemble.predict(X)
        results['ensemble_confidence'] = np.max(ensemble_proba, axis=1)
        
        # Anomaly detection
        results['anomaly_score'] = self.anomaly_detector.predict_proba(X)
        results['is_anomaly'] = self.anomaly_detector.predict(X) == -1
        
        # Combined decision logic
        final_alerts = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            # Case 1: Ensemble says attack with high confidence
            if results['ensemble_prediction'][i] != 'BENIGN' and \
               results['ensemble_prediction'][i] != 0 and \
               results['ensemble_confidence'][i] >= self.ensemble_confidence_threshold:
                final_alerts[i] = 2  # High priority alert
                
            # Case 2: Ensemble says normal, but anomaly detector flags it
            elif results['is_anomaly'][i]:
                final_alerts[i] = 1  # Potential zero-day, needs review
                
            # Case 3: Both agree it's normal
            else:
                final_alerts[i] = 0  # Normal
        
        results['final_alert'] = final_alerts
        
        # Log summary
        n_high = np.sum(final_alerts == 2)
        n_review = np.sum(final_alerts == 1)
        n_normal = np.sum(final_alerts == 0)
        logger.info(f"Detection results: {n_high} high-priority, {n_review} for review, {n_normal} normal")
        
        return results
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate detector performance.
        
        Args:
            X: Test features
            y_true: True labels
            
        Returns:
            Performance metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # For multiclass, get probability of predicted class
        if y_proba.ndim > 1:
            proba_max = np.max(y_proba, axis=1)
        else:
            proba_max = y_proba
        
        # Calculate metrics
        metrics = self.metrics.calculate(y_true, y_pred)
        
        # Print summary
        self.metrics.print_summary()
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save hybrid detector to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            'ensemble': self.ensemble,
            'anomaly_detector': self.anomaly_detector,
            'is_trained': self.is_trained,
            'ensemble_confidence_threshold': self.ensemble_confidence_threshold,
            'anomaly_score_threshold': self.anomaly_score_threshold
        }
        
        joblib.dump(save_data, path)
        logger.info(f"HybridDetector saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'HybridDetector':
        """Load hybrid detector from disk."""
        save_data = joblib.load(path)
        
        detector = cls.__new__(cls)
        detector.name = "HybridDetector"
        detector.ensemble = save_data['ensemble']
        detector.anomaly_detector = save_data['anomaly_detector']
        detector.is_trained = save_data['is_trained']
        detector.ensemble_confidence_threshold = save_data['ensemble_confidence_threshold']
        detector.anomaly_score_threshold = save_data['anomaly_score_threshold']
        detector.metrics = MetricsCalculator()
        
        logger.info(f"HybridDetector loaded from {path}")
        return detector
