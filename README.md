# Pothole Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

Real-time pothole detection using YOLOv8 with a **6-stage detection pipeline**.

---

## 🎯 Features

- **6-Stage Detection Pipeline**: Clean → Find → Track → Isolate → Read → Identify
- **Real-time Detection**: Live camera feed processing
- **Object Tracking**: Track potholes across frames with unique IDs
- **Severity Classification**: LOW, MEDIUM, HIGH based on size and confidence
- **Depth Estimation**: SHALLOW, MODERATE, DEEP
- **Visual Output**: Color-coded bounding boxes and detailed labels

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Detection

```bash
# Run with webcam
python pothole_detector.py

# Specify camera
python pothole_detector.py --camera 0

# Save detections
python pothole_detector.py --save

# Test mode (no camera needed)
python pothole_detector.py --test
```

### Controls

- Press **'q'** to quit
- Press **'s'** to save current frame

---

## 📋 Command Options

| Option | Description |
|--------|-------------|
| `--camera N` | Use camera index N (default: 0) |
| `--model PATH` | YOLO model path (default: yolov8n.pt) |
| `--confidence N` | Detection threshold 0-1 (default: 0.25) |
| `--save` | Save detection frames |
| `--no-tracking` | Disable object tracking |
| `--test` | Test mode without camera |
| `--log-level` | DEBUG, INFO, WARNING, ERROR |

---

## 🔄 6-Stage Pipeline

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

## 📊 Output

### Console Output
```
🚨 POTHOLE | ID:1 | HIGH | Depth:DEEP | Conf:87% | Size:12%
```

### Visual Display
- **Red** = HIGH severity
- **Orange** = MEDIUM severity  
- **Yellow** = LOW severity

### Saved Files
```
results/detections/detection_TIMESTAMP.jpg
```

---

## 📁 Project Structure

```
Real-time-pothole-detection/
├── pothole_detector.py      # ← MAIN FILE - Run this!
├── requirements.txt         # Dependencies
├── yolov8n.pt              # YOLO model
├── config/                  # Configuration
├── scripts/
│   ├── train.py            # Model training
│   └── prepare_dataset.py  # Dataset preparation
├── Datasets/               # Training data
├── models/                 # Trained models
├── logs/                   # Log files
├── results/                # Detection results
└── docs/                   # Documentation
    ├── ARCHITECTURE.md
    ├── CODEBASE_AUDIT_*.md
    └── implementation_plan.md
```

---

## 🛠️ Training Custom Model

### Prepare Dataset

```bash
python scripts/prepare_dataset.py --val-split 0.2
```

### Train Model

```bash
python scripts/train.py --model yolov8n --epochs 100 --batch 16
```

### Use Custom Model

```bash
python pothole_detector.py --model models/weights/best.pt
```

---

## ⚠️ Troubleshooting

### Camera Not Found
```bash
# Try different camera indices
python pothole_detector.py --camera 0
python pothole_detector.py --camera 1
```

### No GPU / Slow Performance
The system will automatically use CPU if CUDA is not available.

### Test Without Camera
```bash
python pothole_detector.py --test
```

---

## 📄 License

MIT License

---

## ✅ Status

**Simplified & Ready to Use!**

One file, one command: `python pothole_detector.py`
