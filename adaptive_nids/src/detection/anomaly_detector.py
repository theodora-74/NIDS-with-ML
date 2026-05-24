#!/usr/bin/env python3
"""
Anomaly Detection for Zero-Day Attack Detection
================================================
Combines Isolation Forest, One-Class SVM, and Autoencoder
for detecting unknown/novel attacks.
Uses PyTorch instead of TensorFlow for better CPU compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple
import warnings

from .base_detector import BaseDetector
from ..utils.logger import setup_logger
from ..utils.config_loader import config

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class AutoencoderPyTorch(nn.Module):
    """
    Autoencoder Neural Network for Anomaly Detection using PyTorch.
    
    Architecture:
    -------------
    Input → Encoder → Bottleneck → Decoder → Output
    
    Anomalies have HIGH reconstruction error because the
    model hasn't learned to reconstruct unusual patterns.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 16, hidden_layers: list = [64, 32]):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Bottleneck layer size
            hidden_layers: Hidden layer sizes
        """
        super().__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for units in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, units))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(units))
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = units
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dim
        for units in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, units))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(units))
            prev_dim = units
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for each sample.
        
        Args:
            x: Input features
            
        Returns:
            Mean squared error per sample
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            x_reconstructed = self.forward(x_tensor)
            mse = torch.mean((x_tensor - x_reconstructed) ** 2, dim=1)
            return mse.numpy()


class AnomalyDetector(BaseDetector):
    """
    Ensemble Anomaly Detector for Zero-Day Attack Detection.
    
    Why Unsupervised Learning?
    --------------------------
    Zero-day attacks are NEW attacks with no training labels.
    We can't train a supervised model without examples!
    
    Solution: Learn what "NORMAL" traffic looks like.
    Anything significantly different = potential zero-day attack.
    
    Our Approach:
    -------------
    1. Isolation Forest: Isolates anomalies by random partitioning
    2. One-Class SVM: Learns a boundary around normal data
    3. Autoencoder: High reconstruction error = anomaly
    
    Final decision: Voting across all three methods
    """
    
    def __init__(self):
        """Initialize anomaly detection ensemble."""
        super().__init__("AnomalyDetector")
        
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.thresholds: Dict[str, float] = {}
        self.autoencoder: Optional[AutoencoderPyTorch] = None
        self.autoencoder_threshold: float = 0.0
        
        # Load configuration
        config.load()
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize anomaly detection models."""
        
        # Isolation Forest
        if_config = config.get('anomaly.isolation_forest', {})
        if if_config.get('enabled', True):
            self.models['isolation_forest'] = IsolationForest(
                n_estimators=if_config.get('n_estimators', 100),
                contamination=if_config.get('contamination', 0.1),
                max_samples=if_config.get('max_samples', 0.8),
                random_state=42,
                n_jobs=-1
            )
            logger.debug("Isolation Forest initialized")
        
        # One-Class SVM
        svm_config = config.get('anomaly.one_class_svm', {})
        if svm_config.get('enabled', True):
            self.models['one_class_svm'] = OneClassSVM(
                kernel=svm_config.get('kernel', 'rbf'),
                nu=svm_config.get('nu', 0.1),
                gamma='scale'
            )
            logger.debug("One-Class SVM initialized")
        
        logger.info(f"Anomaly detector initialized with {len(self.models)} models + Autoencoder")
    
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Train anomaly detectors on NORMAL traffic only.
        
        Args:
            X: Training features
            y: Labels (optional) - if provided, trains only on normal class
        """
        logger.info("="*50)
        logger.info("TRAINING ANOMALY DETECTOR")
        logger.info("="*50)
        
        # Filter to normal traffic only if labels provided
        if y is not None:
            # Convert to string array for comparison
            y_str = np.array([str(label) for label in y])
            
            # Check for normal labels (case-insensitive)
            normal_labels = ['benign', 'normal', '0']
            normal_mask = np.array([str(label).lower() in normal_labels for label in y])
            
            X_normal = X[normal_mask]
            logger.info(f"Training on {len(X_normal)} normal samples (out of {len(X)})")
            
            # If no normal samples found, use all data (fallback)
            if len(X_normal) == 0:
                logger.warning("No normal samples found! Using all data for anomaly baseline.")
                X_normal = X
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_normal)
        
        # Train traditional models
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_scaled)
                logger.info(f"  ✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"  ✗ {name} training failed: {e}")
        
        # Train Autoencoder
        ae_config = config.get('anomaly.autoencoder', {})
        if ae_config.get('enabled', True):
            logger.info("Training Autoencoder (PyTorch)...")
            self._train_autoencoder(X_scaled, ae_config)
        
        self.is_trained = True
        logger.info("="*50)
        logger.info("ANOMALY DETECTOR TRAINING COMPLETE")
        logger.info("="*50)
    
    def _train_autoencoder(self, X: np.ndarray, ae_config: Dict) -> None:
        """
        Train autoencoder neural network with PyTorch.
        
        Args:
            X: Scaled normal traffic features
            ae_config: Autoencoder configuration
        """
        input_dim = X.shape[1]
        encoding_dim = ae_config.get('encoding_dim', 16)
        hidden_layers = ae_config.get('hidden_layers', [64, 32])
        epochs = ae_config.get('epochs', 50)
        batch_size = ae_config.get('batch_size', 256)
        threshold_percentile = ae_config.get('threshold_percentile', 95)
        
        # Create autoencoder
        self.autoencoder = AutoencoderPyTorch(input_dim, encoding_dim, hidden_layers)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.autoencoder.train()
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                reconstructed = self.autoencoder(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.debug(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Calculate threshold from reconstruction errors on normal data
        self.autoencoder.eval()
        reconstruction_errors = self.autoencoder.get_reconstruction_error(X)
        self.autoencoder_threshold = np.percentile(reconstruction_errors, threshold_percentile)
        
        logger.info(f"  ✓ Autoencoder trained (threshold: {self.autoencoder_threshold:.6f})")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Detect anomalies using voting.
        
        Args:
            X: Input features
            
        Returns:
            Array of -1 (anomaly) or 1 (normal)
        """
        if not self.is_trained:
            raise RuntimeError("Anomaly detector not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        votes = np.zeros(X.shape[0])
        n_models = 0
        
        # Traditional models vote
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)  # Returns 1 (normal) or -1 (anomaly)
                votes += (pred == -1).astype(int)  # Count anomaly votes
                n_models += 1
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
        
        # Autoencoder votes
        if self.autoencoder is not None:
            reconstruction_errors = self.autoencoder.get_reconstruction_error(X_scaled)
            ae_anomalies = (reconstruction_errors > self.autoencoder_threshold).astype(int)
            votes += ae_anomalies
            n_models += 1
        
        # Majority voting: anomaly if more than half vote anomaly
        threshold = n_models / 2
        predictions = np.where(votes > threshold, -1, 1)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly probability scores.
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_trained:
            raise RuntimeError("Anomaly detector not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        scores = np.zeros(X.shape[0])
        n_models = 0
        
        # Isolation Forest score
        if 'isolation_forest' in self.models:
            # score_samples returns negative (more anomalous = more negative)
            if_scores = -self.models['isolation_forest'].score_samples(X_scaled)
            if_min, if_max = if_scores.min(), if_scores.max()
            if if_max - if_min > 1e-8:
                scores += (if_scores - if_min) / (if_max - if_min)
            n_models += 1
        
        # One-Class SVM score
        if 'one_class_svm' in self.models:
            # decision_function returns signed distance (negative = anomaly)
            svm_scores = -self.models['one_class_svm'].decision_function(X_scaled)
            svm_min, svm_max = svm_scores.min(), svm_scores.max()
            if svm_max - svm_min > 1e-8:
                scores += (svm_scores - svm_min) / (svm_max - svm_min)
            n_models += 1
        
        # Autoencoder score
        if self.autoencoder is not None:
            ae_scores = self.autoencoder.get_reconstruction_error(X_scaled)
            ae_min, ae_max = ae_scores.min(), ae_scores.max()
            if ae_max - ae_min > 1e-8:
                scores += (ae_scores - ae_min) / (ae_max - ae_min)
            n_models += 1
        
        # Average and normalize to [0, 1]
        if n_models > 0:
            scores = scores / n_models
        
        return scores
    
    def get_anomaly_details(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get detailed anomaly scores from each model.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with scores from each model
        """
        X_scaled = self.scaler.transform(X)
        details = {}
        
        if 'isolation_forest' in self.models:
            details['isolation_forest'] = -self.models['isolation_forest'].score_samples(X_scaled)
        
        if 'one_class_svm' in self.models:
            details['one_class_svm'] = -self.models['one_class_svm'].decision_function(X_scaled)
        
        if self.autoencoder is not None:
            details['autoencoder'] = self.autoencoder.get_reconstruction_error(X_scaled)
            details['autoencoder_threshold'] = self.autoencoder_threshold
        
        return details
