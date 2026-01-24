"""
YOLOv5 Pothole Detection Model Training Script
================================================
This script trains a YOLOv5 model for pothole detection with severity classification.
The model will be deployed on Raspberry Pi for real-time detection.

Features:
- Multi-class detection (pothole severity: low, medium, high)
- Data augmentation for robust detection
- Transfer learning from pretrained YOLOv5
- Model optimization for edge deployment
"""

import os
import yaml
import torch
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
class Config:
    """Training configuration parameters"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATASET_ROOT = PROJECT_ROOT / "Datasets"
    IMAGES_DIR = DATASET_ROOT / "images"
    LABELS_DIR = DATASET_ROOT / "labels"
    
    # YOLOv5 settings
    YOLOV5_REPO = "ultralytics/yolov5"
    MODEL_SIZE = "yolov5s"  # Options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
    
    # Training parameters
    IMG_SIZE = 640
    BATCH_SIZE = 16  # Adjust based on GPU memory
    EPOCHS = 100
    WORKERS = 4
    
    # Class names (severity levels)
    CLASSES = ['low', 'medium', 'high']
    NUM_CLASSES = len(CLASSES)
    
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1
    
    # Output directories
    RUNS_DIR = PROJECT_ROOT / "runs"
    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    
    # Device
    DEVICE = '0' if torch.cuda.is_available() else 'cpu'
    
    # Hyperparameters
    LEARNING_RATE = 0.01
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    
    # Early stopping
    PATIENCE = 50  # Epochs to wait for improvement
    
    # Augmentation (YOLOv5 built-in)
    AUGMENT = True
    
    # Model optimization for Raspberry Pi
    OPTIMIZE_FOR_EDGE = True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.RUNS_DIR.mkdir(exist_ok=True)
        cls.WEIGHTS_DIR.mkdir(exist_ok=True)
        
        # Create dataset structure
        for split in ['train', 'val', 'test']:
            (cls.DATASET_ROOT / split / 'images').mkdir(parents=True, exist_ok=True)
            (cls.DATASET_ROOT / split / 'labels').mkdir(parents=True, exist_ok=True)


def check_environment():
    """Check and display environment information"""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: Training on CPU will be significantly slower!")
    
    print(f"Device: {Config.DEVICE}")
    print("=" * 70)
    print()


def prepare_dataset():
    """
    Prepare dataset in YOLO format and split into train/val/test
    
    Expected label format (YOLO):
    <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]
    """
    print("=" * 70)
    print("DATASET PREPARATION")
    print("=" * 70)
    
    import random
    from glob import glob
    
    # Get all image files
    image_files = list(Config.IMAGES_DIR.glob("*.jpg"))
    print(f"Total images found: {len(image_files)}")
    
    # Check for corresponding label files
    label_files = []
    for img_path in image_files:
        label_path = Config.LABELS_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            label_files.append(label_path)
    
    print(f"Total labels found: {len(label_files)}")
    
    if len(label_files) == 0:
        print("\nWARNING: No label files found!")
        print("Please ensure labels are in YOLO format in the 'labels' directory.")
        print("Label format: <class_id> <x_center> <y_center> <width> <height>")
        return False
    
    # Create paired list of images and labels
    paired_data = []
    for img_path in image_files:
        label_path = Config.LABELS_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            paired_data.append((img_path, label_path))
    
    print(f"Total paired samples: {len(paired_data)}")
    
    # Shuffle data
    random.seed(42)
    random.shuffle(paired_data)
    
    # Calculate split indices
    total = len(paired_data)
    train_end = int(total * Config.TRAIN_RATIO)
    val_end = train_end + int(total * Config.VAL_RATIO)
    
    # Split data
    train_data = paired_data[:train_end]
    val_data = paired_data[train_end:val_end]
    test_data = paired_data[val_end:]
    
    print(f"\nDataset Split:")
    print(f"  Train: {len(train_data)} samples ({Config.TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {len(val_data)} samples ({Config.VAL_RATIO*100:.0f}%)")
    print(f"  Test:  {len(test_data)} samples ({Config.TEST_RATIO*100:.0f}%)")
    
    # Copy files to respective directories
    def copy_split(data, split_name):
        for img_path, label_path in data:
            # Copy image
            dst_img = Config.DATASET_ROOT / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Copy label
            dst_label = Config.DATASET_ROOT / split_name / 'labels' / label_path.name
            shutil.copy2(label_path, dst_label)
    
    print("\nCopying files to split directories...")
    copy_split(train_data, 'train')
    copy_split(val_data, 'val')
    copy_split(test_data, 'test')
    
    print("Dataset preparation complete!")
    print("=" * 70)
    print()
    
    return True


def create_dataset_yaml():
    """Create YAML configuration file for YOLOv5"""
    print("Creating dataset configuration file...")
    
    dataset_config = {
        'path': str(Config.DATASET_ROOT.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': Config.NUM_CLASSES,
        'names': Config.CLASSES
    }
    
    yaml_path = Config.DATASET_ROOT / 'pothole_dataset.yaml'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset config saved to: {yaml_path}")
    print()
    
    return yaml_path


def clone_yolov5():
    """Clone YOLOv5 repository if not exists"""
    import subprocess
    yolov5_dir = Config.PROJECT_ROOT / 'yolov5'
    
    if not yolov5_dir.exists():
        print("Cloning YOLOv5 repository...")
        subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5.git', str(yolov5_dir)])
        print("Installing YOLOv5 requirements...")
        subprocess.run(['pip', 'install', '-r', str(yolov5_dir / 'requirements.txt')])
    else:
        print("YOLOv5 repository already exists.")
    
    return yolov5_dir


def train_model(dataset_yaml_path):
    """
    Train YOLOv5 model
    """
    print("=" * 70)
    print("STARTING MODEL TRAINING")
    print("=" * 70)
    
    yolov5_dir = clone_yolov5()
    
    # Training command - use subprocess for better path handling
    import subprocess
    
    train_cmd = [
        'python', str(yolov5_dir / 'train.py'),
        '--img', str(Config.IMG_SIZE),
        '--batch', str(Config.BATCH_SIZE),
        '--epochs', str(Config.EPOCHS),
        '--data', str(dataset_yaml_path),
        '--weights', f'{Config.MODEL_SIZE}.pt',
        '--project', str(Config.RUNS_DIR),
        '--name', f'pothole_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        '--device', Config.DEVICE,
        '--workers', str(Config.WORKERS),
        '--patience', str(Config.PATIENCE),
        '--cache',  # Cache images for faster training
    ]
    
    if Config.AUGMENT:
        train_cmd.append('--augment')
    
    # Execute training
    print("\nTraining Command:")
    print(' '.join(f'"{arg}"' if ' ' in str(arg) else str(arg) for arg in train_cmd))
    print("\nStarting training... This may take several hours.")
    print("=" * 70)
    print()
    
    subprocess.run(train_cmd)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


def optimize_for_raspberry_pi(weights_path):
    """
    Optimize trained model for Raspberry Pi deployment
    - Convert to TorchScript
    - Quantization (optional)
    - Export to ONNX
    """
    print("=" * 70)
    print("OPTIMIZING MODEL FOR RASPBERRY PI")
    print("=" * 70)
    
    yolov5_dir = Config.PROJECT_ROOT / 'yolov5'
    
    import subprocess
    
    # Export to TorchScript
    print("\n1. Exporting to TorchScript...")
    export_cmd = [
        'python', str(yolov5_dir / 'export.py'),
        '--weights', str(weights_path),
        '--include', 'torchscript',
        '--device', 'cpu',  # Export for CPU (Raspberry Pi)
        '--simplify',
    ]
    subprocess.run(export_cmd)
    
    # Export to ONNX (better for edge devices)
    print("\n2. Exporting to ONNX...")
    export_cmd = [
        'python', str(yolov5_dir / 'export.py'),
        '--weights', str(weights_path),
        '--include', 'onnx',
        '--device', 'cpu',
        '--simplify',
    ]
    subprocess.run(export_cmd)
    
    print("\nModel optimization complete!")
    print("Optimized models saved in the same directory as weights.")
    print("=" * 70)


def main():
    """Main training pipeline"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  POTHOLE DETECTION MODEL TRAINING PIPELINE".center(68) + "*")
    print("*" + "  Multi-Sensor Embedded System for Real-Time Detection".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    # Step 1: Check environment
    check_environment()
    
    # Step 2: Create directories
    print("Creating project directories...")
    Config.create_directories()
    print("Directories created successfully!\n")
    
    # Step 3: Prepare dataset
    dataset_ready = prepare_dataset()
    
    if not dataset_ready:
        print("\nERROR: Dataset preparation failed!")
        print("Please ensure your labels are properly formatted and try again.")
        return
    
    # Step 4: Create dataset YAML
    dataset_yaml_path = create_dataset_yaml()
    
    # Step 5: Train model
    print("\nReady to start training!")
    print(f"Model: {Config.MODEL_SIZE}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Classes: {Config.CLASSES}")
    print()
    
    response = input("Start training? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        train_model(dataset_yaml_path)
        
        # Step 6: Find best weights
        print("\nLooking for best trained weights...")
        runs_dir = Config.RUNS_DIR
        latest_run = max(runs_dir.glob('pothole_detection_*'), key=os.path.getmtime)
        best_weights = latest_run / 'weights' / 'best.pt'
        
        if best_weights.exists():
            print(f"Best weights found: {best_weights}")
            
            # Step 7: Optimize for Raspberry Pi
            if Config.OPTIMIZE_FOR_EDGE:
                response = input("\nOptimize model for Raspberry Pi? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    optimize_for_raspberry_pi(best_weights)
            
            # Copy to weights directory
            final_weights = Config.WEIGHTS_DIR / 'pothole_best.pt'
            shutil.copy2(best_weights, final_weights)
            print(f"\nFinal weights saved to: {final_weights}")
        
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE COMPLETE!")
        print("=" * 70)
        print("\nNext Steps:")
        print("1. Review training metrics in the runs directory")
        print("2. Test the model using test_model.py")
        print("3. Deploy to Raspberry Pi for real-time detection")
        print("=" * 70)
    else:
        print("Training cancelled.")


if __name__ == "__main__":
    main()
