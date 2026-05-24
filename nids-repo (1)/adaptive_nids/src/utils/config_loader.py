#!/usr/bin/env python3
"""
Configuration Loader for Adaptive ML-NIDS
=========================================
Loads and validates YAML configuration files.
"""

import yaml
import os
from typing import Dict, Any, Optional
from .logger import setup_logger

logger = setup_logger(__name__)


class ConfigLoader:
    """Load and manage configuration settings."""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        """Singleton pattern - only one config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, config_path: str = "/opt/adaptive_nids/config/config.yaml") -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return self._config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., 'ensemble.models.xgboost.weight')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return full configuration dictionary."""
        return self._config


# Global config instance
config = ConfigLoader()
