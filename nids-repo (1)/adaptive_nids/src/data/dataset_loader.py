#!/usr/bin/env python3
"""
Dataset Loader for ML-NIDS
==========================
Loads and preprocesses standard IDS datasets.
Supports: CICIDS2017, NSL-KDD, UNSW-NB15
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
import warnings

from ..utils.logger import setup_logger
from ..utils.config_loader import config

logger = setup_logger(__name__)
warnings.filterwarnings('ignore')


class DatasetLoader:
    """
    Load and preprocess standard IDS benchmark datasets.
    
    Supported Datasets:
    -------------------
    1. CICIDS2017 - Canadian Institute for Cybersecurity
       - Attack types: DoS, DDoS, Brute Force, Botnet, Infiltration, Web Attack
       - ~3 million samples
       
    2. NSL-KDD - Improved KDD Cup 99
       - Attack types: DoS, Probe, R2L, U2R
       - ~150K samples
       
    3. UNSW-NB15 - Australian Centre for Cyber Security
       - Attack types: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, etc.
       - ~2.5 million samples
    """
    
    def __init__(self):
        """Initialize dataset loader."""
        config.load()
        self.data_dir = config.get('paths.data_dir', '/opt/adaptive_nids/data')
        
    def load_cicids2017(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load CICIDS2017 dataset.
        
        Expected file structure:
        /data/datasets/CICIDS2017/
            ├── Friday-WorkingHours-Afternoon-DDos.csv
            ├── Friday-WorkingHours-Afternoon-PortScan.csv
            ├── ...
        
        Args:
            data_path: Path to CICIDS2017 folder
            
        Returns:
            Combined DataFrame
        """
        if data_path is None:
            data_path = config.get('datasets.cicids2017', 
                                   os.path.join(self.data_dir, 'datasets/CICIDS2017'))
        
        logger.info(f"Loading CICIDS2017 from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CICIDS2017 not found at {data_path}")
        
        # Find all CSV files
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        # Load and combine
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(data_path, csv_file)
            logger.info(f"  Loading {csv_file}...")
            
            try:
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"  Could not load {csv_file}: {e}")
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Clean column names (remove leading/trailing spaces)
        combined.columns = combined.columns.str.strip()
        
        # Standardize label column
        label_cols = ['Label', 'label', ' Label']
        for col in label_cols:
            if col in combined.columns:
                combined = combined.rename(columns={col: 'Label'})
                break
        
        # Basic cleaning
        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = combined.dropna(subset=['Label'])
        
        logger.info(f"CICIDS2017 loaded: {len(combined)} samples")
        logger.info(f"Attack distribution:\n{combined['Label'].value_counts()}")
        
        return combined
    
    def load_nslkdd(self, data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load NSL-KDD dataset.
        
        Expected file structure:
        /data/datasets/NSL-KDD/
            ├── KDDTrain+.txt
            └── KDDTest+.txt
        
        Args:
            data_path: Path to NSL-KDD folder
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if data_path is None:
            data_path = config.get('datasets.nslkdd',
                                   os.path.join(self.data_dir, 'datasets/NSL-KDD'))
        
        logger.info(f"Loading NSL-KDD from {data_path}")
        
        # NSL-KDD column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'Label', 'difficulty'
        ]
        
        # Load training data
        train_path = os.path.join(data_path, 'KDDTrain+.txt')
        if os.path.exists(train_path):
            train_df = pd.read_csv(train_path, names=columns, header=None)
            train_df = train_df.drop(columns=['difficulty'])
            logger.info(f"NSL-KDD Train loaded: {len(train_df)} samples")
        else:
            raise FileNotFoundError(f"KDDTrain+.txt not found in {data_path}")
        
        # Load test data
        test_path = os.path.join(data_path, 'KDDTest+.txt')
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path, names=columns, header=None)
            test_df = test_df.drop(columns=['difficulty'])
            logger.info(f"NSL-KDD Test loaded: {len(test_df)} samples")
        else:
            test_df = None
            logger.warning("KDDTest+.txt not found")
        
        return train_df, test_df
    
    def load_unswnb15(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load UNSW-NB15 dataset.
        
        Args:
            data_path: Path to UNSW-NB15 folder
            
        Returns:
            Combined DataFrame
        """
        if data_path is None:
            data_path = config.get('datasets.unswnb15',
                                   os.path.join(self.data_dir, 'datasets/UNSW-NB15'))
        
        logger.info(f"Loading UNSW-NB15 from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"UNSW-NB15 not found at {data_path}")
        
        # Find CSV files
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        dfs = []
        for csv_file in csv_files:
            file_path = os.path.join(data_path, csv_file)
            logger.info(f"  Loading {csv_file}...")
            try:
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"  Could not load {csv_file}: {e}")
        
        if not dfs:
            raise FileNotFoundError(f"No CSV files could be loaded from {data_path}")
        
        combined = pd.concat(dfs, ignore_index=True)
        
        # Standardize label column
        if 'attack_cat' in combined.columns:
            combined = combined.rename(columns={'attack_cat': 'Label'})
        elif 'label' in combined.columns:
            combined = combined.rename(columns={'label': 'Label'})
        
        logger.info(f"UNSW-NB15 loaded: {len(combined)} samples")
        
        return combined
    
    def generate_synthetic(self, n_samples: int = 100000, attack_ratio: float = 0.3) -> pd.DataFrame:
        """
        Generate synthetic network traffic data for testing.
        
        Args:
            n_samples: Number of samples to generate
            attack_ratio: Proportion of attack samples
            
        Returns:
            Synthetic DataFrame
        """
        logger.info(f"Generating {n_samples} synthetic samples...")
        
        np.random.seed(42)
        
        n_attacks = int(n_samples * attack_ratio)
        n_normal = n_samples - n_attacks
        
        # Generate normal traffic
        normal_data = {
            'duration': np.random.exponential(5, n_normal),
            'packet_count': np.random.poisson(50, n_normal),
            'total_bytes': np.random.lognormal(8, 1, n_normal),
            'avg_packet_size': np.random.normal(500, 100, n_normal),
            'packet_rate': np.random.exponential(10, n_normal),
            'byte_rate': np.random.lognormal(10, 1, n_normal),
            'flag_SYN': np.random.poisson(2, n_normal),
            'flag_ACK': np.random.poisson(30, n_normal),
            'flag_FIN': np.random.poisson(1, n_normal),
            'flag_RST': np.random.poisson(0.1, n_normal),
            'src_port': np.random.randint(1024, 65535, n_normal),
            'dst_port': np.random.choice([80, 443, 22, 25, 53, 3306], n_normal),
            'Label': ['BENIGN'] * n_normal
        }
        
        # Generate attack traffic (different patterns)
        attack_types = ['DoS', 'DDoS', 'PortScan', 'BruteForce']
        attack_labels = np.random.choice(attack_types, n_attacks)
        
        attack_data = {
            'duration': np.random.exponential(1, n_attacks),  # Shorter
            'packet_count': np.random.poisson(500, n_attacks),  # More packets
            'total_bytes': np.random.lognormal(12, 2, n_attacks),  # More bytes
            'avg_packet_size': np.random.normal(200, 50, n_attacks),  # Smaller packets
            'packet_rate': np.random.exponential(100, n_attacks),  # Higher rate
            'byte_rate': np.random.lognormal(13, 1.5, n_attacks),  # Higher rate
            'flag_SYN': np.random.poisson(50, n_attacks),  # More SYN
            'flag_ACK': np.random.poisson(10, n_attacks),  # Less ACK
            'flag_FIN': np.random.poisson(0.5, n_attacks),
            'flag_RST': np.random.poisson(5, n_attacks),  # More RST
            'src_port': np.random.randint(1, 1024, n_attacks),  # Lower ports
            'dst_port': np.random.choice([22, 23, 3389, 445], n_attacks),
            'Label': attack_labels
        }
        
        # Combine
        normal_df = pd.DataFrame(normal_data)
        attack_df = pd.DataFrame(attack_data)
        combined = pd.concat([normal_df, attack_df], ignore_index=True)
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Synthetic data generated: {len(combined)} samples")
        logger.info(f"Distribution:\n{combined['Label'].value_counts()}")
        
        return combined
    
    def prepare_train_test(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        label_column: str = 'Label'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion for testing
            label_column: Label column name
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size,
            stratify=df[label_column],
            random_state=42
        )
        
        logger.info(f"Train/Test split: {len(train_df)}/{len(test_df)}")
        
        return train_df, test_df
