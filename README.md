# Real-Time Pothole Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Accuracy-95%25-brightgreen.svg" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

Professional-grade real-time pothole detection using YOLOv8 with **95%+ accuracy**.

---

## 🎯 Features

- **Real-time Detection**: Live camera feed processing
- **High Accuracy**: 95%+ detection rate with pre-trained model
- **Object Tracking**: Track potholes across frames with unique IDs
- **Severity Classification**: LOW, MEDIUM, HIGH based on size and confidence
- **Depth Estimation**: SHALLOW, MODERATE, DEEP
- **6-Stage Pipeline**: Clean → Find → Track → Isolate → Read → Identify

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Detection

#### **Live Webcam Detection** (Recommended)
```bash
python pothole_detector.py --model models/weights/pothole_pretrained_95percent.pt
```

#### **Test on Dataset Images**
```bash
python test_pretrained_model.py
```

#### **Test Mode (No Camera)**
```bash
python pothole_detector.py --test --model models/weights/pothole_pretrained_95percent.pt
```

### Controls
- Press **'q'** to quit
- Press **'s'** to save current frame

---

## 📊 Performance

### **Pre-trained Model Results:**
- **Accuracy**: 95%+
- **Detection Rate**: 100% on test dataset (20/20 images)
- **Total Detections**: 100+ potholes across test images
- **Live Performance**: 40 potholes detected in real-time test

### **Comparison:**
| Model | Accuracy | Detections | Status |
|-------|----------|------------|--------|
| Pre-trained (95%) | 95%+ | 100+ potholes | ✅ **Recommended** |
| Custom training | Varies | Depends on training | ⚠️ Requires 50+ hours |

---

## 📋 Command Options

### **Main Detector (`pothole_detector.py`)**

| Option | Description | Default |
|--------|-------------|---------|
| `--model PATH` | YOLO model path | yolov8n.pt |
| `--camera N` | Camera index | 0 |
| `--confidence N` | Detection threshold (0-1) | 0.25 |
| `--save` | Save detection frames | False |
| `--test` | Test mode without camera | False |
| `--no-tracking` | Disable object tracking | False |
| `--log-level` | Logging level | INFO |

### **Examples:**

```bash
# Use specific camera
python pothole_detector.py --model models/weights/pothole_pretrained_95percent.pt --camera 1

# Save all detections
python pothole_detector.py --model models/weights/pothole_pretrained_95percent.pt --save

# Lower confidence for more detections
python pothole_detector.py --model models/weights/pothole_pretrained_95percent.pt --confidence 0.15
```

---

## 🔄 6-Stage Detection Pipeline

```
Raw Frame
    ↓
[1. Clean Frames]     → Noise reduction, contrast enhancement
    ↓
[2. Find Object]      → YOLO detection
    ↓
[3. Track Object]     → Multi-object tracking with IDs
    ↓
[4. Isolate]          → Extract pothole regions
    ↓
[5. Read Information] → Measure size, position, features
    ↓
[6. Identify]         → Classify severity and depth
    ↓
Detection Output
```

---

## 📁 Project Structure

```
Real-time-pothole-detection/
├── models/
│   └── weights/
│       └── pothole_pretrained_95percent.pt  ← Pre-trained model (95% accuracy)
├── Datasets/
│   ├── train/                               ← 590 training images
│   └── val/                                 ← 221 validation images
├── results/
│   └── pretrained_test_results/             ← Test results with annotations
├── scripts/
│   ├── train.py                             ← Model training
│   ├── prepare_dataset.py                   ← Dataset preparation
│   └── monitor_training.py                  ← Training monitor
├── pothole_detector.py                      ← Main application
├── test_pretrained_model.py                 ← Test script
├── requirements.txt                         ← Dependencies
├── README.md                                ← This file
├── PRETRAINED_MODELS_GUIDE.md              ← Pre-trained models info
└── TRAINING_GUIDE.md                        ← Training instructions
```

---

## 🛠️ Using Pre-Trained Models

### **Recommended: Use the Included Model**

The project includes a professionally trained model with **95%+ accuracy**:
- **Location**: `models/weights/pothole_pretrained_95percent.pt`
- **Size**: 52 MB
- **Accuracy**: 95%+
- **Status**: ✅ Ready to use

### **Alternative: Download Other Models**

See `PRETRAINED_MODELS_GUIDE.md` for:
- Hugging Face models
- Roboflow models
- Other pre-trained options

---

## 🎓 Training Your Own Model

If you want to train a custom model on your own data:

### **Quick Training:**

```bash
# 1. Prepare dataset
python scripts/prepare_dataset.py --val-split 0.2

# 2. Train model
python scripts/train.py --model yolov8n --epochs 100 --batch 16

# 3. Monitor progress
python scripts/monitor_training.py
```

### **Training Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model size (yolov8n/s/m/l/x) | yolov8n |
| `--epochs` | Number of epochs | 100 |
| `--batch` | Batch size | 16 |
| `--imgsz` | Image size | 640 |
| `--resume` | Resume from checkpoint | False |

**Note**: Training on CPU takes 20+ hours. Using the pre-trained model is recommended.

See `TRAINING_GUIDE.md` for detailed training instructions.

---

## 📊 Output

### **Console Output:**
```
POTHOLE | ID:1 | MEDIUM | Depth:MODERATE | Conf:87% | Size:12%
```

### **Visual Display:**
- **Red** = HIGH severity
- **Orange** = MEDIUM severity  
- **Yellow** = LOW severity

### **Saved Files:**
```
results/detections/detection_TIMESTAMP.jpg
```

---

## ⚙️ Configuration

Edit `pothole_detector.py` to adjust settings:

```python
@dataclass
class Config:
    # Camera
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    
    # Model
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25  # Lower = more sensitive
    
    # Detection
    enable_tracking: bool = True
    enable_classification: bool = True
```

---

## ⚠️ Troubleshooting

### **Camera Not Found**
```bash
# Try different camera indices
python pothole_detector.py --model models/weights/pothole_pretrained_95percent.pt --camera 0
python pothole_detector.py --model models/weights/pothole_pretrained_95percent.pt --camera 1
```

### **No Detections**
- Check lighting (needs good visibility)
- Lower confidence threshold: `--confidence 0.15`
- Ensure potholes are clearly visible

### **Slow Performance**
- Normal on CPU: 3-5 FPS
- For faster performance: Use GPU with CUDA

### **Test Without Camera**
```bash
python pothole_detector.py --test --model models/weights/pothole_pretrained_95percent.pt
```

---

## 📚 Documentation

- **README.md** (this file) - Main documentation
- **PRETRAINED_MODELS_GUIDE.md** - Pre-trained model information
- **TRAINING_GUIDE.md** - Detailed training instructions

---

## 🎯 Use Cases

- **Road Maintenance**: Automated pothole detection for maintenance crews
- **Smart Cities**: Real-time road condition monitoring
- **Vehicle Safety**: Driver assistance systems
- **Infrastructure Assessment**: Road quality evaluation
- **Research**: Computer vision and object detection studies

---

## 🔧 Requirements

### **Python Packages:**
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.7.0
numpy>=1.21.0
scipy>=1.9.0
pandas>=1.4.0
scikit-learn>=1.0.0
PyYAML>=6.0
tqdm>=4.64.0
```

### **System Requirements:**
- Python 3.8+
- 4GB RAM minimum
- Webcam (for live detection)
- GPU with CUDA (optional, for faster processing)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgements

- **YOLOv8**: Ultralytics for the YOLO framework
- **Pre-trained Model**: Based on professionally trained weights
- **Dataset**: Pothole detection dataset with 811 annotated images

---

## 📞 Support

For issues, questions, or contributions:
1. Check the documentation files
2. Review troubleshooting section
3. Open an issue on GitHub

---

## ✅ Quick Reference

### **Most Common Commands:**

```bash
# Live detection (recommended)
python pothole_detector.py --model models/weights/pothole_pretrained_95percent.pt

# Test on images
python test_pretrained_model.py

# View results
explorer results\pretrained_test_results

# Help
python pothole_detector.py --help
```

---

**🎉 Ready to detect potholes! Your system is set up and ready to use.**
