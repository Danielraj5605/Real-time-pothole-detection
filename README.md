# Real-Time Pothole Detection System

An integrated multi-sensor embedded system for real-time pothole detection, severity classification, and driver alerting using Raspberry Pi.

## 🎯 Project Overview

This project implements a **Hierarchical Multimodal Edge Fusion Architecture** that combines:
- **Camera-based Vision** (YOLOv5) for visual pothole detection
- **Accelerometer Data** (MPU6050) for vibration sensing
- **GPS Tagging** (NEO-6M) for geo-location
- **Real-time Alerts** for approaching vehicles

### Key Features

✅ **Multi-sensor Fusion**: Combines camera and accelerometer data for robust detection  
✅ **Severity Classification**: Categorizes potholes as Low, Medium, or High severity  
✅ **GPS Geo-tagging**: Records exact location of detected potholes  
✅ **Real-time Alerts**: Warns drivers of high-severity potholes ahead  
✅ **Edge Computing**: Runs on Raspberry Pi 4 for practical deployment  
✅ **Low Cost**: Uses affordable, readily available components  

---

## 📁 Project Structure

```
Real-time-pothole-detection/
├── Datasets/
│   ├── images/          # Raw images (2009 images)
│   ├── labels/          # YOLO format annotations
│   ├── train/           # Training split
│   ├── val/             # Validation split
│   └── test/            # Test split
├── runs/                # Training runs and logs
├── weights/             # Trained model weights
├── yolov5/              # YOLOv5 repository (auto-cloned)
├── train_yolov5.py      # Main training script
├── test_model.py        # Model testing and evaluation
├── prepare_labels.py    # Label validation and visualization
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository (if not already done)
cd "d:\Personal Projects\Real-time-pothole-detection"

# Install dependencies
pip install -r requirements.txt

# The YOLOv5 repository will be auto-cloned during training
```

### 2. Prepare Your Dataset

**Option A: If you already have YOLO format labels**

```bash
# Validate your labels
python prepare_labels.py --mode validate

# Visualize some samples
python prepare_labels.py --mode visualize
```

**Option B: If you need to create labels**

1. Use annotation tools:
   - [LabelImg](https://github.com/heartexlabs/labelImg) (Desktop)
   - [Roboflow](https://roboflow.com/) (Web-based)
   - [CVAT](https://www.cvat.ai/) (Advanced)

2. Export in **YOLO format**

3. Place labels in `Datasets/labels/` directory

4. Label format example:
   ```
   # <class_id> <x_center> <y_center> <width> <height>
   0 0.5 0.5 0.3 0.2    # Low severity pothole
   1 0.7 0.3 0.15 0.15  # Medium severity
   2 0.2 0.8 0.25 0.18  # High severity
   ```

### 3. Train the Model

```bash
# Start training
python train_yolov5.py
```

**Training Configuration:**
- Model: YOLOv5s (small, fast, suitable for Raspberry Pi)
- Image Size: 640x640
- Batch Size: 16 (adjust based on GPU memory)
- Epochs: 100
- Classes: 3 (low, medium, high severity)

**Expected Training Time:**
- With GPU (RTX 3060): ~2-3 hours
- With CPU: ~12-24 hours

### 4. Test the Model

```bash
# Test on a single image
python test_model.py --mode single --image path/to/image.jpg

# Evaluate on test set
python test_model.py --mode test_set

# Benchmark inference speed
python test_model.py --mode benchmark
```

---

## 📊 Dataset Information

- **Total Images**: 2,009
- **Image Format**: JPG
- **Image Sources**: 
  - Real-world pothole images
  - Video frame extractions
  - Various lighting and weather conditions

**Data Split:**
- Training: 70% (1,406 images)
- Validation: 20% (402 images)
- Test: 10% (201 images)

---

## 🎓 Model Training Details

### Architecture: YOLOv5s

- **Backbone**: CSPDarknet53
- **Neck**: PANet
- **Head**: YOLOv5 Detection Head
- **Parameters**: ~7.2M
- **GFLOPs**: ~16.5

### Training Hyperparameters

```python
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
```

### Data Augmentation

- Mosaic augmentation
- Random scaling
- Random cropping
- Color jittering
- Horizontal flipping

### Performance Metrics

After training, check:
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision at IoU 0.5 to 0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

---

## 🔧 Hardware Requirements

### For Training

**Minimum:**
- CPU: Intel i5 or equivalent
- RAM: 8GB
- Storage: 20GB free space
- GPU: Optional but recommended (NVIDIA with CUDA support)

**Recommended:**
- CPU: Intel i7/AMD Ryzen 7 or better
- RAM: 16GB+
- GPU: NVIDIA RTX 3060 or better
- Storage: SSD with 50GB+ free space

### For Deployment (Raspberry Pi)

**Hardware:**
- Raspberry Pi 4 (4GB or 8GB RAM)
- Camera Module (Pi Camera v2 or USB webcam)
- MPU6050 Accelerometer/Gyroscope
- NEO-6M GPS Module
- MicroSD Card (32GB+ Class 10)
- Power Supply (5V 3A)
- Optional: Buzzer/LED for alerts

---

## 📈 Expected Results

### Detection Performance

- **Accuracy**: 85-95% (depends on dataset quality)
- **Inference Speed**:
  - Desktop GPU: 30-60 FPS
  - Raspberry Pi 4: 2-5 FPS (YOLOv5s)
  - Raspberry Pi 4: 5-10 FPS (YOLOv5n)

### Severity Classification

The model classifies potholes into three categories:

| Severity | Description | Visual Indicator |
|----------|-------------|------------------|
| **Low** | Minor surface damage | 🟢 Green |
| **Medium** | Moderate damage, caution needed | 🟠 Orange |
| **High** | Severe damage, immediate attention | 🔴 Red |

---

## 🎯 Next Steps

### 1. Model Optimization for Raspberry Pi

```bash
# After training, optimize the model
python train_yolov5.py
# Follow prompts to optimize for edge deployment
```

This will create:
- `best.torchscript` - TorchScript format
- `best.onnx` - ONNX format (recommended for Pi)

### 2. Raspberry Pi Deployment

1. **Install Dependencies on Pi:**
   ```bash
   pip install opencv-python-headless
   pip install onnxruntime
   pip install numpy
   ```

2. **Transfer Model:**
   ```bash
   scp weights/best.onnx pi@raspberrypi.local:~/pothole_detection/
   ```

3. **Run Inference:**
   - Use the optimized ONNX model
   - Integrate with camera feed
   - Add GPS tagging
   - Implement alert system

### 3. System Integration

- Integrate accelerometer data (MPU6050)
- Add GPS module (NEO-6M)
- Implement sensor fusion logic
- Create local alert server
- Test complete system

---

## 📝 Usage Examples

### Training

```bash
# Basic training
python train_yolov5.py

# Custom configuration (edit Config class in script)
# - Change MODEL_SIZE to 'yolov5n' for faster inference
# - Adjust BATCH_SIZE based on GPU memory
# - Modify EPOCHS for longer/shorter training
```

### Testing

```bash
# Single image inference
python test_model.py --mode single --image test.jpg

# Batch evaluation
python test_model.py --mode test_set --test_dir Datasets/test/images

# Speed benchmark
python test_model.py --mode benchmark
```

### Label Validation

```bash
# Validate all labels
python prepare_labels.py --mode validate

# Visualize specific image
python prepare_labels.py --mode visualize --image image_name.jpg

# Check label format
python prepare_labels.py --mode check --label_file path/to/label.txt

# Show sample format
python prepare_labels.py --mode sample
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in train_yolov5.py
BATCH_SIZE = 8  # or 4
```

**2. No Labels Found**
```bash
# Check label format and location
python prepare_labels.py --mode validate
```

**3. Slow Training**
```bash
# Use smaller model
MODEL_SIZE = "yolov5n"  # Nano version
```

**4. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## 📚 References

### YOLOv5
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv5 Documentation](https://docs.ultralytics.com/)

### Research Papers
- "You Only Look Once: Unified, Real-Time Object Detection" (YOLO)
- "YOLOv5: Improvements and Applications"

### Datasets
- Custom pothole dataset (2,009 images)
- Various road conditions and lighting

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Better data augmentation strategies
- Multi-modal sensor fusion algorithms
- Real-time alert optimization
- Mobile app integration
- Cloud-based pothole mapping

---

## 📄 License

This project is for educational and research purposes.

---

## 👥 Authors

Real-Time Pothole Detection System  
Multi-Sensor Embedded System Project

---

## 🙏 Acknowledgments

- YOLOv5 by Ultralytics
- PyTorch team
- OpenCV community
- Raspberry Pi Foundation

---

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review training logs in `runs/` directory
3. Validate dataset using `prepare_labels.py`

---

**Happy Training! 🚀**
