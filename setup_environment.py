"""
Environment Setup and Validation Script
========================================
Checks system requirements and prepares environment for training
"""

import sys
import subprocess
import platform
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_python_version():
    """Check Python version"""
    print_header("Python Version Check")
    
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False


def check_pip():
    """Check if pip is available"""
    print_header("Package Manager Check")
    
    try:
        result = subprocess.run(['pip', '--version'], 
                              capture_output=True, text=True, timeout=5)
        print(f"✅ pip is installed: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"❌ pip is not available: {e}")
        return False


def check_pytorch():
    """Check PyTorch installation"""
    print_header("PyTorch Check")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("   Training will be significantly slower")
        
        return True
    except ImportError:
        print("❌ PyTorch is not installed")
        print("\nInstall with:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False


def check_opencv():
    """Check OpenCV installation"""
    print_header("OpenCV Check")
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV is not installed")
        print("\nInstall with:")
        print("  pip install opencv-python")
        return False


def check_dependencies():
    """Check all required dependencies"""
    print_header("Dependency Check")
    
    required = {
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'pandas': 'pandas',
        'seaborn': 'seaborn'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall all dependencies with:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def check_disk_space():
    """Check available disk space"""
    print_header("Disk Space Check")
    
    import shutil
    
    project_root = Path(__file__).parent
    stat = shutil.disk_usage(project_root)
    
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    used_gb = stat.used / (1024**3)
    
    print(f"Total: {total_gb:.2f} GB")
    print(f"Used: {used_gb:.2f} GB")
    print(f"Free: {free_gb:.2f} GB")
    
    if free_gb >= 20:
        print(f"✅ Sufficient disk space ({free_gb:.2f} GB free)")
        return True
    else:
        print(f"⚠️  Low disk space ({free_gb:.2f} GB free)")
        print("   Recommended: At least 20 GB free")
        return False


def check_dataset():
    """Check dataset availability"""
    print_header("Dataset Check")
    
    project_root = Path(__file__).parent
    images_dir = project_root / "Datasets" / "images"
    labels_dir = project_root / "Datasets" / "labels"
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not labels_dir.exists():
        print(f"⚠️  Labels directory not found: {labels_dir}")
        print("   You will need to create annotations")
        labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {labels_dir}")
    
    # Count images
    images = list(images_dir.glob("*.jpg"))
    print(f"✅ Found {len(images)} images")
    
    # Count labels
    labels = list(labels_dir.glob("*.txt"))
    print(f"   Found {len(labels)} label files")
    
    if len(images) == 0:
        print("❌ No images found in dataset")
        return False
    
    if len(labels) == 0:
        print("⚠️  No labels found - you need to annotate your images")
        print("\nOptions:")
        print("  1. Use LabelImg: pip install labelImg")
        print("  2. Use Roboflow: https://roboflow.com")
        print("  3. Use CVAT: https://www.cvat.ai")
        return False
    
    coverage = (len(labels) / len(images)) * 100
    print(f"   Label coverage: {coverage:.1f}%")
    
    if coverage < 50:
        print("⚠️  Less than 50% of images are labeled")
        print("   Recommended: Label at least 80% of images")
    
    return True


def check_git():
    """Check if git is available"""
    print_header("Git Check")
    
    try:
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True, timeout=5)
        print(f"✅ Git is installed: {result.stdout.strip()}")
        return True
    except Exception:
        print("❌ Git is not installed")
        print("   Git is required to clone YOLOv5 repository")
        print("\nDownload from: https://git-scm.com/downloads")
        return False


def create_directories():
    """Create necessary project directories"""
    print_header("Directory Setup")
    
    project_root = Path(__file__).parent
    
    directories = [
        "Datasets/images",
        "Datasets/labels",
        "Datasets/train/images",
        "Datasets/train/labels",
        "Datasets/val/images",
        "Datasets/val/labels",
        "Datasets/test/images",
        "Datasets/test/labels",
        "runs",
        "weights"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")
    
    print("\nAll directories created successfully!")
    return True


def install_dependencies():
    """Install dependencies from requirements.txt"""
    print_header("Installing Dependencies")
    
    project_root = Path(__file__).parent
    requirements_file = project_root / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("Installing packages from requirements.txt...")
    print("This may take several minutes...\n")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 
                       str(requirements_file)], check=True)
        print("\n✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        return False


def print_summary(results):
    """Print summary of checks"""
    print("\n" + "=" * 70)
    print("  SETUP SUMMARY")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to start training!")
        print("\nNext steps:")
        print("  1. Validate your dataset: python prepare_labels.py --mode validate")
        print("  2. Start training: python train_yolov5.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before training.")
    
    print("=" * 70)


def main():
    """Main setup function"""
    print("\n" + "*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  POTHOLE DETECTION - ENVIRONMENT SETUP".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    print("\nThis script will check your system and prepare the environment")
    print("for training the pothole detection model.\n")
    
    # Run checks
    results = {
        "Python Version": check_python_version(),
        "Package Manager (pip)": check_pip(),
        "PyTorch": check_pytorch(),
        "OpenCV": check_opencv(),
        "Dependencies": check_dependencies(),
        "Disk Space": check_disk_space(),
        "Git": check_git(),
        "Dataset": check_dataset(),
        "Directories": create_directories()
    }
    
    # Print summary
    print_summary(results)
    
    # Offer to install dependencies if missing
    if not results["Dependencies"]:
        print("\n" + "=" * 70)
        response = input("Would you like to install missing dependencies now? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            if install_dependencies():
                print("\n✅ Setup complete! Please run this script again to verify.")
            else:
                print("\n❌ Installation failed. Please install manually:")
                print("   pip install -r requirements.txt")
        print("=" * 70)


if __name__ == "__main__":
    main()
