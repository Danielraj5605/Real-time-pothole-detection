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
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config, get_logger, setup_logger
from src.vision import VisionTrainer


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
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger = setup_logger(
        "training",
        level="INFO",
        log_file=str(log_dir / "training.log")
    )
    
    print("="*60)
    print("🎯 YOLOV8 POTHOLE DETECTION TRAINING")
    print("="*60)
    
    # Validate dataset
    data_yaml = project_root / args.data
    if not data_yaml.exists():
        print(f"\n❌ Dataset config not found: {data_yaml}")
        print("\n💡 Run 'python scripts/prepare_dataset.py' first!")
        return 1
    
    print(f"\n📊 Dataset: {data_yaml}")
    print(f"🔧 Model: {args.model}")
    print(f"📈 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch}")
    print(f"🖼️  Image size: {args.imgsz}")
    
    # Initialize trainer
    print("\n🚀 Initializing trainer...")
    trainer = VisionTrainer(model_type=args.model)
    
    # Print model info
    info = trainer.get_model_info()
    print(f"   Parameters: {info['parameters']:,}")
    
    # Train
    print("\n🏋️ Starting training...")
    print("-"*60)
    
    try:
        metrics = trainer.train(
            data_yaml=str(data_yaml),
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.imgsz,
            project=str(project_root / "models/yolo_training"),
            name=args.name,
            resume=args.resume
        )
        
        print("-"*60)
        print("\n✅ Training Complete!")
        print(f"\n📊 Results:")
        print(f"   Train Accuracy: {metrics.get('train_accuracy', 'N/A')}")
        print(f"   Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
        
        # Export best weights
        weights_dir = project_root / "models/weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        export_path = weights_dir / f"pothole_{args.model}_best.pt"
        trainer.export(str(export_path))
        
        print(f"\n💾 Best weights: {export_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1
    
    print("\n🎉 Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
