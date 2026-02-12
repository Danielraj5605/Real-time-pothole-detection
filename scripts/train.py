#!/usr/bin/env python3
"""
YOLOv8 Training Script for Pothole Detection

Trains a YOLOv8 model on the prepared pothole dataset.

Usage:
    python scripts/train.py                    # Default training
    python scripts/train.py --epochs 50        # Custom epochs
    python scripts/train.py --model yolov8s    # Different model size
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logger(name: str, log_file: str = None, level: str = "INFO"):
    """Setup logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Pothole Detection"
    )
    parser.add_argument(
        '--model', type=str, default='yolov8n',
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        help='Model size (n=nano, s=small, m=medium, l=large, x=extra)'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch', type=int, default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz', type=int, default=640,
        help='Image size'
    )
    parser.add_argument(
        '--data', type=str,
        default='Datasets/pothole_dataset.yaml',
        help='Path to dataset YAML'
    )
    parser.add_argument(
        '--name', type=str, default=None,
        help='Run name (auto-generated if not specified)'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--device', type=str, default='',
        help='Device to use (e.g., 0, 1, cpu). Empty string for auto-detect'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        "training",
        level="INFO",
        log_file=str(log_dir / f"training_{timestamp}.log")
    )
    
    print("="*60)
    print("YOLOV8 POTHOLE DETECTION TRAINING")
    print("="*60)
    
    # Validate dataset
    data_yaml = project_root / args.data
    if not data_yaml.exists():
        print(f"\n[ERROR] Dataset config not found: {data_yaml}")
        print("\n[TIP] Run 'python scripts/prepare_dataset.py' first!")
        return 1
    
    print(f"\nDataset: {data_yaml}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    
    # Import YOLO
    try:
        from ultralytics import YOLO
        print("\n[OK] Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"\n[ERROR] Failed to import YOLO: {e}")
        print("[TIP] Install with: pip install ultralytics")
        return 1
    
    # Initialize model
    print(f"\nInitializing {args.model} model...")
    try:
        # Check if we have a pretrained base model
        base_model_path = project_root / f"{args.model}.pt"
        if base_model_path.exists():
            print(f"   Loading from: {base_model_path}")
            model = YOLO(str(base_model_path))
        else:
            print(f"   Downloading pretrained {args.model} weights...")
            model = YOLO(f"{args.model}.pt")
        
        # Print model info
        print(f"   Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        print(f"\n[ERROR] Model initialization failed: {e}")
        return 1
    
    # Setup training directory
    training_dir = project_root / "models" / "yolo_training"
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run name if not provided
    if args.name is None:
        args.name = f"pothole_{args.model}_{timestamp}"
    
    # Train
    print("\nStarting training...")
    print("-"*60)
    logger.info(f"Training started: {args.name}")
    
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=str(training_dir),
            name=args.name,
            resume=args.resume,
            device=args.device if args.device else None,
            verbose=True,
            # Training hyperparameters
            patience=50,  # Early stopping patience
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            # Augmentation
            hsv_h=0.015,  # Image HSV-Hue augmentation
            hsv_s=0.7,    # Image HSV-Saturation augmentation
            hsv_v=0.4,    # Image HSV-Value augmentation
            degrees=0.0,  # Image rotation (+/- deg)
            translate=0.1,  # Image translation (+/- fraction)
            scale=0.5,    # Image scale (+/- gain)
            shear=0.0,    # Image shear (+/- deg)
            perspective=0.0,  # Image perspective (+/- fraction)
            flipud=0.0,   # Image flip up-down (probability)
            fliplr=0.5,   # Image flip left-right (probability)
            mosaic=1.0,   # Image mosaic (probability)
        )
        
        print("-"*60)
        print("\n[SUCCESS] Training Complete!")
        
        # Get metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\nFinal Metrics:")
            if 'metrics/mAP50(B)' in metrics:
                print(f"   mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
            if 'metrics/mAP50-95(B)' in metrics:
                print(f"   mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
        
        # Export best weights
        weights_dir = project_root / "models" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the best weights from training
        best_weights = training_dir / args.name / "weights" / "best.pt"
        if best_weights.exists():
            export_path = weights_dir / f"pothole_{args.model}_best.pt"
            
            # Copy the best weights
            import shutil
            shutil.copy(str(best_weights), str(export_path))
            
            print(f"\nBest weights saved to: {export_path}")
            print(f"   Original weights: {best_weights}")
            
            logger.info(f"Training completed successfully. Best weights: {export_path}")
        else:
            print(f"\n[WARNING] Best weights not found at {best_weights}")
            print(f"   Check training directory: {training_dir / args.name}")
        
        # Print training results location
        print(f"\nTraining results: {training_dir / args.name}")
        print(f"   - Weights: {training_dir / args.name / 'weights'}")
        print(f"   - Plots: {training_dir / args.name}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"\n[ERROR] Training failed: {e}")
        return 1
    
    print("\nDone!")
    print("\nNext steps:")
    print(f"   1. Review training plots in: {training_dir / args.name}")
    print(f"   2. Test the model: python pothole_detector.py --test --model models/weights/pothole_{args.model}_best.pt")
    print(f"   3. Run live detection: python pothole_detector.py --model models/weights/pothole_{args.model}_best.pt")
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
