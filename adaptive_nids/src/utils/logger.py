#!/usr/bin/env python3
"""
Logging Configuration for Adaptive ML-NIDS
==========================================
Provides colored console output and file logging.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import colorlog


def setup_logger(name: str, log_dir: str = "/opt/adaptive_nids/logs") -> logging.Logger:
    """
    Configure and return a logger with console and file handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory if not exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_format)
    
    # File handler with rotation
    log_file = os.path.join(log_dir, f"nids_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Create default logger for package
logger = setup_logger("ML-NIDS")
