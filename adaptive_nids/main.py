#!/usr/bin/env python3
"""
Adaptive ML-NIDS Main Application
=================================
Main entry point for training, testing, and running the IDS.
"""

import argparse
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import setup_logger
from src.utils.config_loader import config
from src.data.dataset_loader import DatasetLoader
from src.features.feature_processor import FeatureProcessor
from src.detection.hybrid_detector import HybridDetector

logger = setup_logger("Main")


def train_model(args):
    """Train the hybrid detection model."""
    logger.info("="*60)
    logger.info("      ADAPTIVE ML-NIDS TRAINING")
    logger.info("="*60)
    
    # Load configuration
    config.load()
    
    # Initialize components
    data_loader = DatasetLoader()
    feature_processor = FeatureProcessor()
    detector = HybridDetector()
    
    # Load data
    if args.synthetic:
        logger.info("\nUsing SYNTHETIC data for demo...")
        df = data_loader.generate_synthetic(n_samples=args.samples)
    elif args.dataset == 'cicids2017':
        df = data_loader.load_cicids2017(args.data_path)
    elif args.dataset == 'nslkdd':
        train_df, _ = data_loader.load_nslkdd(args.data_path)
        df = train_df
    elif args.dataset == 'unswnb15':
        df = data_loader.load_unswnb15(args.data_path)
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        return
    
    # Split data
    train_df, test_df = data_loader.prepare_train_test(df, test_size=0.2)
    
    # Process features
    logger.info("\nProcessing features...")
    X_train, y_train = feature_processor.fit_transform(train_df)
    X_test, y_test = feature_processor.transform(test_df)
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # Train model
    detector.train(X_train, y_train)
    
    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("      EVALUATION RESULTS")
    logger.info("="*60)
    
    metrics = detector.evaluate(X_test, y_test)
    
    # Save model
    model_path = os.path.join(config.get('paths.models_dir'), 'hybrid_detector.joblib')
    detector.save(model_path)
    
    # Save feature processor
    processor_path = os.path.join(config.get('paths.models_dir'), 'feature_processor.joblib')
    import joblib
    joblib.dump(feature_processor, processor_path)
    logger.info(f"Feature processor saved to {processor_path}")
    
    logger.info("\n" + "="*60)
    logger.info("      TRAINING COMPLETE!")
    logger.info("="*60)
    
    return metrics


def test_model(args):
    """Test the trained model."""
    logger.info("="*60)
    logger.info("      TESTING ML-NIDS")
    logger.info("="*60)
    
    import joblib
    
    # Load configuration
    config.load()
    
    # Load model and processor
    model_path = os.path.join(config.get('paths.models_dir'), 'hybrid_detector.joblib')
    processor_path = os.path.join(config.get('paths.models_dir'), 'feature_processor.joblib')
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Train first!")
        return
    
    if not os.path.exists(processor_path):
        logger.error(f"Feature processor not found at {processor_path}. Train first!")
        return
    
    detector = HybridDetector.load(model_path)
    feature_processor = joblib.load(processor_path)
    
    # Load test data
    data_loader = DatasetLoader()
    
    if args.synthetic:
        df = data_loader.generate_synthetic(n_samples=10000)
    elif args.dataset == 'cicids2017':
        df = data_loader.load_cicids2017(args.data_path)
    elif args.dataset == 'nslkdd':
        train_df, test_df = data_loader.load_nslkdd(args.data_path)
        df = test_df if test_df is not None else train_df
        logger.info(f"Loaded NSL-KDD test data: {len(df)} samples")
    elif args.dataset == 'unswnb15':
        df = data_loader.load_unswnb15(args.data_path)
    else:
        logger.error("Specify dataset (--dataset) or use --synthetic")
        return
    
    # Process features
    logger.info("Processing features...")
    X_test, y_test = feature_processor.transform(df)
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Run detection
    logger.info("Running detection...")
    results = detector.detect_with_anomaly(X_test)
    
    # Calculate metrics if labels available
    if y_test is not None:
        logger.info("\nCalculating metrics...")
        metrics = detector.evaluate(X_test, y_test)
    
    # Print summary
    print("\n" + "="*50)
    print("      DETECTION SUMMARY")
    print("="*50)
    print(f"  Total samples:     {len(X_test)}")
    print(f"  High-priority:     {np.sum(results['final_alert'] == 2)}")
    print(f"  Need review:       {np.sum(results['final_alert'] == 1)}")
    print(f"  Normal:            {np.sum(results['final_alert'] == 0)}")
    print("="*50)


def demo_mode(args):
    """Quick demonstration with synthetic data."""
    logger.info("="*60)
    logger.info("      ADAPTIVE ML-NIDS DEMO MODE")
    logger.info("="*60)
    
    args.synthetic = True
    args.samples = 50000
    args.dataset = None
    args.data_path = None
    
    train_model(args)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Adaptive ML-NIDS - Machine Learning Network Intrusion Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train --synthetic              # Train with synthetic data
  python main.py train --dataset cicids2017     # Train with CICIDS2017
  python main.py test --synthetic               # Test with synthetic data
  python main.py test --dataset nslkdd          # Test with NSL-KDD
  python main.py demo                           # Quick demo mode
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', choices=['cicids2017', 'nslkdd', 'unswnb15'],
                             help='Dataset to use')
    train_parser.add_argument('--data-path', help='Path to dataset')
    train_parser.add_argument('--synthetic', action='store_true',
                             help='Use synthetic data')
    train_parser.add_argument('--samples', type=int, default=100000,
                             help='Number of synthetic samples')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--dataset', choices=['cicids2017', 'nslkdd', 'unswnb15'],
                            help='Dataset to use')
    test_parser.add_argument('--data-path', help='Path to dataset')
    test_parser.add_argument('--synthetic', action='store_true',
                            help='Use synthetic data')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Quick demonstration')
    
    args = parser.parse_args()
    
    # Load config
    config.load()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'test':
        test_model(args)
    elif args.command == 'demo':
        demo_mode(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
