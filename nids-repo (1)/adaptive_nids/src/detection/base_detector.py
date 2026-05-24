#!/usr/bin/env python3
"""
Base Detector Interface
=======================
Abstract base class for all detection models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
import joblib
import os

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseDetector(ABC):
    """Abstract base class for detection models."""
    
    def __init__(self, name: str):
        """
        Initialize detector.
        
        Args:
            name: Detector name for logging/saving
        """
        self.name = name
        self.is_trained = False
        self.model = None
        
    @abstractmethod
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Train the detector.
        
        Args:
            X: Training features
            y: Training labels (optional for unsupervised)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability array
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Save path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"{self.name} saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseDetector':
        """
        Load model from disk.
        
        Args:
            path: Model path
            
        Returns:
            Loaded detector
        """
        detector = joblib.load(path)
        logger.info(f"Loaded detector from {path}")
        return detector
