#!/usr/bin/env python3
"""
Feature Processor for Network Traffic Data
===========================================
Handles preprocessing, scaling, and feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Optional, Tuple, List
import warnings

from ..utils.logger import setup_logger
from ..utils.config_loader import config

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class FeatureProcessor:
    """
    Process and engineer features for ML-NIDS.
    
    Pipeline:
    ---------
    1. Handle missing values
    2. Remove infinite values
    3. Encode categorical features
    4. Scale numerical features
    5. Feature selection (optional)
    """
    
    def __init__(self):
        """Initialize feature processor."""
        config.load()
        
        self.scaler: Optional[Any] = None
        self.imputer: Optional[SimpleImputer] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.is_fitted = False
        
        # Configure scaler based on config
        scaling_method = config.get('features.scaling', 'standard')
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Configure imputer
        missing_strategy = config.get('features.handle_missing', 'median')
        self.imputer = SimpleImputer(strategy=missing_strategy)
        
        logger.info(f"FeatureProcessor initialized (scaling: {scaling_method})")
    
    def fit(self, df: pd.DataFrame, label_column: str = 'Label') -> 'FeatureProcessor':
        """
        Fit processor on training data.
        
        Args:
            df: Training DataFrame
            label_column: Name of label column
            
        Returns:
            Self for chaining
        """
        logger.info("Fitting FeatureProcessor...")
        
        # Separate features and labels
        if label_column in df.columns:
            X = df.drop(columns=[label_column])
        else:
            X = df.copy()
        
        # Identify column types
        self.numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.feature_names = self.numeric_columns + self.categorical_columns
        
        logger.info(f"  Numeric columns: {len(self.numeric_columns)}")
        logger.info(f"  Categorical columns: {len(self.categorical_columns)}")
        
        # Fit label encoders for categorical columns
        for col in self.categorical_columns:
            le = LabelEncoder()
            # Handle unknown values by fitting on all unique values including a placeholder
            unique_vals = X[col].dropna().unique().tolist()
            le.fit(unique_vals)
            self.label_encoders[col] = le
        
        # Fit imputer and scaler on numeric data
        if self.numeric_columns:
            numeric_data = X[self.numeric_columns].replace([np.inf, -np.inf], np.nan)
            self.imputer.fit(numeric_data)
            imputed_data = self.imputer.transform(numeric_data)
            self.scaler.fit(imputed_data)
        
        self.is_fitted = True
        logger.info("FeatureProcessor fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame, label_column: str = 'Label') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted processor.
        
        Args:
            df: Input DataFrame
            label_column: Name of label column
            
        Returns:
            Tuple of (features, labels) or (features, None) if no labels
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureProcessor not fitted. Call fit() first.")
        
        # Separate features and labels
        y = None
        if label_column in df.columns:
            y = df[label_column].values
            X = df.drop(columns=[label_column])
        else:
            X = df.copy()
        
        # Process categorical columns
        for col in self.categorical_columns:
            if col in X.columns:
                le = self.label_encoders[col]
                # Handle unseen categories
                X[col] = X[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Process numeric columns
        if self.numeric_columns:
            # Get numeric data that exists in input
            existing_numeric = [c for c in self.numeric_columns if c in X.columns]
            
            if existing_numeric:
                numeric_data = X[existing_numeric].replace([np.inf, -np.inf], np.nan)
                imputed_data = self.imputer.transform(numeric_data)
                scaled_data = self.scaler.transform(imputed_data)
                X[existing_numeric] = scaled_data
        
        # Ensure column order matches training
        X = X.reindex(columns=self.feature_names, fill_value=0)
        
        return X.values.astype(np.float32), y
    
    def fit_transform(self, df: pd.DataFrame, label_column: str = 'Label') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit processor and transform data.
        
        Args:
            df: Input DataFrame
            label_column: Name of label column
            
        Returns:
            Tuple of (features, labels)
        """
        self.fit(df, label_column)
        return self.transform(df, label_column)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names in order."""
        return self.feature_names


class NetworkFeatureExtractor:
    """
    Extract features from raw network traffic.
    
    Features Extracted:
    -------------------
    - Flow duration
    - Packet count (forward/backward)
    - Byte count (forward/backward)
    - Packet length statistics
    - Inter-arrival time statistics
    - Flag counts (SYN, ACK, FIN, RST, PSH, URG)
    - Flow ratios
    """
    
    @staticmethod
    def extract_flow_features(packets: list) -> Dict[str, float]:
        """
        Extract features from a flow of packets.
        
        Args:
            packets: List of packet dictionaries
            
        Returns:
            Dictionary of features
        """
        if not packets:
            return {}
        
        features = {}
        
        # Basic counts
        features['packet_count'] = len(packets)
        features['total_bytes'] = sum(p.get('length', 0) for p in packets)
        
        # Duration
        timestamps = [p.get('timestamp', 0) for p in packets]
        if len(timestamps) > 1:
            features['duration'] = max(timestamps) - min(timestamps)
        else:
            features['duration'] = 0
        
        # Packet length statistics
        lengths = [p.get('length', 0) for p in packets]
        features['avg_packet_size'] = np.mean(lengths)
        features['std_packet_size'] = np.std(lengths)
        features['min_packet_size'] = np.min(lengths)
        features['max_packet_size'] = np.max(lengths)
        
        # Inter-arrival times
        if len(timestamps) > 1:
            iat = np.diff(sorted(timestamps))
            features['avg_iat'] = np.mean(iat)
            features['std_iat'] = np.std(iat)
            features['min_iat'] = np.min(iat)
            features['max_iat'] = np.max(iat)
        else:
            features['avg_iat'] = 0
            features['std_iat'] = 0
            features['min_iat'] = 0
            features['max_iat'] = 0
        
        # Rates
        if features['duration'] > 0:
            features['packet_rate'] = features['packet_count'] / features['duration']
            features['byte_rate'] = features['total_bytes'] / features['duration']
        else:
            features['packet_rate'] = features['packet_count']
            features['byte_rate'] = features['total_bytes']
        
        # TCP flags (if available)
        flags = {'SYN': 0, 'ACK': 0, 'FIN': 0, 'RST': 0, 'PSH': 0, 'URG': 0}
        for p in packets:
            for flag in flags:
                if p.get(flag, False):
                    flags[flag] += 1
        features.update({f'flag_{k}': v for k, v in flags.items()})
        
        return features
