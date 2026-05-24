#!/usr/bin/env python3
"""
Ensemble Classifier for Known Attack Detection
===============================================
Combines XGBoost, LightGBM, Random Forest, and Gradient Boosting
using weighted voting for robust classification.
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, Optional, Tuple
import warnings

from .base_detector import BaseDetector
from ..utils.logger import setup_logger
from ..utils.config_loader import config

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class EnsembleClassifier(BaseDetector):
    """
    Weighted Ensemble Classifier for Network Intrusion Detection.
    
    Architecture:
    -------------
    ┌─────────────────────────────────────────────────────────────┐
    │                    INPUT FEATURES                            │
    └─────────────────────────────────────────────────────────────┘
                              │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ XGBoost  │      │ LightGBM │      │ Random   │      │ Gradient │
    │  (30%)   │      │  (30%)   │      │ Forest   │      │ Boosting │
    │          │      │          │      │  (25%)   │      │  (15%)   │
    └──────────┘      └──────────┘      └──────────┘      └──────────┘
          │                  │                  │                │
          └──────────────────┼──────────────────┘                │
                             │                                    │
                             ▼                                    │
                    ┌─────────────────┐                           │
                    │ Weighted Voting │◄──────────────────────────┘
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Final Prediction│
                    └─────────────────┘
    """
    
    def __init__(self):
        """Initialize ensemble with configured models."""
        super().__init__("EnsembleClassifier")
        
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.label_encoder = LabelEncoder()
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        
        # Load configuration
        config.load()
        
        self._initialize_models()
        
    def _initialize_models(self) -> None:
        """Initialize all ensemble models with configured hyperparameters."""
        
        # XGBoost - Gradient Boosting with regularization
        xgb_config = config.get('ensemble.models.xgboost', {})
        if xgb_config.get('enabled', True):
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=xgb_config.get('n_estimators', 200),
                max_depth=xgb_config.get('max_depth', 10),
                learning_rate=xgb_config.get('learning_rate', 0.1),
                subsample=xgb_config.get('subsample', 0.8),
                colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
                objective='multi:softprob',
                eval_metric='mlogloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            self.weights['xgboost'] = xgb_config.get('weight', 0.30)
            logger.debug("XGBoost initialized")
        
        # LightGBM - Fast gradient boosting
        lgb_config = config.get('ensemble.models.lightgbm', {})
        if lgb_config.get('enabled', True):
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=lgb_config.get('n_estimators', 200),
                max_depth=lgb_config.get('max_depth', 10),
                learning_rate=lgb_config.get('learning_rate', 0.1),
                subsample=lgb_config.get('subsample', 0.8),
                colsample_bytree=lgb_config.get('colsample_bytree', 0.8),
                objective='multiclass',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            self.weights['lightgbm'] = lgb_config.get('weight', 0.30)
            logger.debug("LightGBM initialized")
        
        # Random Forest - Bagging ensemble
        rf_config = config.get('ensemble.models.random_forest', {})
        if rf_config.get('enabled', True):
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=rf_config.get('n_estimators', 150),
                max_depth=rf_config.get('max_depth', 15),
                min_samples_split=rf_config.get('min_samples_split', 5),
                min_samples_leaf=rf_config.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
            self.weights['random_forest'] = rf_config.get('weight', 0.25)
            logger.debug("Random Forest initialized")
        
        # Gradient Boosting - Sequential ensemble
        gb_config = config.get('ensemble.models.gradient_boosting', {})
        if gb_config.get('enabled', True):
            self.models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=gb_config.get('n_estimators', 100),
                max_depth=gb_config.get('max_depth', 8),
                learning_rate=gb_config.get('learning_rate', 0.1),
                subsample=gb_config.get('subsample', 0.8),
                random_state=42
            )
            self.weights['gradient_boosting'] = gb_config.get('weight', 0.15)
            logger.debug("Gradient Boosting initialized")
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.info(f"Ensemble initialized with {len(self.models)} models")
        logger.info(f"Model weights: {self.weights}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train all ensemble models.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        logger.info("="*50)
        logger.info("TRAINING ENSEMBLE CLASSIFIER")
        logger.info("="*50)
        
        # Encode labels if needed
        if y.dtype == object or isinstance(y[0], str):
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
        else:
            y_encoded = y
            self.classes_ = np.unique(y)
        
        self.n_classes_ = len(self.classes_)
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Number of classes: {self.n_classes_}")
        logger.info(f"Class distribution: {np.bincount(y_encoded.astype(int))}")
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X, y_encoded)
                logger.info(f"  ✓ {name} trained successfully")
            except Exception as e:
                logger.error(f"  ✗ {name} training failed: {e}")
                raise
        
        self.is_trained = True
        logger.info("="*50)
        logger.info("ENSEMBLE TRAINING COMPLETE")
        logger.info("="*50)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get weighted probability predictions from ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Weighted average probabilities (n_samples, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained. Call train() first.")
        
        # Collect probabilities from each model
        all_probas = []
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X)
                all_probas.append((name, proba))
            except Exception as e:
                logger.warning(f"Could not get probabilities from {name}: {e}")
        
        if not all_probas:
            raise RuntimeError("No models could produce predictions")
        
        # Weighted average
        weighted_proba = np.zeros((X.shape[0], self.n_classes_))
        for name, proba in all_probas:
            weighted_proba += self.weights[name] * proba
        
        return weighted_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make final predictions using weighted voting.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        
        # Convert back to original labels if label encoder was used
        if self.label_encoder.classes_ is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, np.ndarray]:
        """
        Get feature importance from each model.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of feature importances per model
        """
        importances = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
        
        # Average importance across models
        if importances:
            avg_importance = np.mean(list(importances.values()), axis=0)
            importances['ensemble_average'] = avg_importance
            
            # Log top features
            if feature_names:
                top_idx = np.argsort(avg_importance)[-10:][::-1]
                logger.info("Top 10 important features:")
                for idx in top_idx:
                    logger.info(f"  {feature_names[idx]}: {avg_importance[idx]:.4f}")
        
        return importances
