# Multimodal Pothole Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A production-grade **offline multimodal ML pipeline** for pothole detection and severity classification. Combines computer vision (YOLOv8) with accelerometer signal processing for robust, real-world pothole detection.

---

## 🎯 Features

- **Vision Pipeline**: YOLOv8-based pothole detection with configurable models
- **Accelerometer Pipeline**: Signal processing with sliding windows and severity classification
- **Multimodal Fusion**: Rule-based and ML-based fusion strategies
- **Complete Offline System**: No internet required for inference
- **Production Ready**: Modular architecture, comprehensive logging, and configuration management

---

## 📁 Project Structure

```
Real-time-pothole-detection/
│
├── config/
│   └── config.yaml              # Master configuration file
│
├── src/
│   ├── __init__.py
│   │
│   ├── vision/                  # Computer vision pipeline
│   │   ├── detector.py          # YOLOv8 inference
│   │   ├── trainer.py           # Model training
│   │   └── features.py          # Feature extraction
│   │
│   ├── accelerometer/           # Accelerometer pipeline
│   │   ├── processor.py         # Signal processing
│   │   ├── features.py          # Feature extraction
│   │   └── classifier.py        # Severity classification
│   │
│   ├── fusion/                  # Multimodal fusion
│   │   ├── engine.py            # Fusion engine
│   │   ├── rules.py             # Rule-based fusion
│   │   └── alerts.py            # Alert management
│   │
│   └── utils/                   # Shared utilities
│       ├── config_loader.py     # Configuration management
│       └── logger.py            # Logging utilities
│
├── scripts/
│   ├── train.py                 # Training script
│   ├── prepare_dataset.py       # Dataset preparation
│   └── evaluate.py              # Evaluation script
│
├── Datasets/                    # Dataset storage
│   ├── Pothole_Image_Data/      # Pothole images
│   ├── Pothole/                 # Accelerometer CSV data
│   └── images/                  # Additional images
│
├── models/                      # Trained models
│   └── weights/                 # Model weights
│
├── demo.py                      # Demo application
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Real-time-pothole-detection.git
cd Real-time-pothole-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo

```bash
python demo.py
```

This will:
- Initialize all pipelines (vision, accelerometer, fusion)
- Run detection on sample data
- Display results and generate visualizations

### 3. Train Custom Model

```bash
# Prepare dataset
python scripts/prepare_dataset.py

# Train YOLOv8
python scripts/train.py --epochs 100 --model yolov8n
```

---

## 📊 Pipelines

### Vision Pipeline (YOLOv8)

Detects potholes in images using state-of-the-art object detection.

```python
from src.vision import PotholeDetector

detector = PotholeDetector("models/weights/pothole_best.pt")
detections = detector.detect("road_image.jpg")

for det in detections:
    print(f"Pothole: conf={det.confidence:.2f}, area={det.area}")
```

**Features Extracted:**
- Detection confidence
- Bounding box area (normalized)
- Aspect ratio
- Number of detections

### Accelerometer Pipeline

Processes accelerometer data with sliding windows for severity classification.

```python
from src.accelerometer import AccelerometerProcessor, AccelFeatureExtractor

processor = AccelerometerProcessor(window_size=50)
extractor = AccelFeatureExtractor()

for window in processor.process_file("trip_sensors.csv"):
    features = extractor.extract(window)
    print(f"Peak: {features.peak_acceleration:.2f}g, RMS: {features.rms_vibration:.2f}g")
```

**Features Extracted:**
- Peak acceleration (X, Y, Z, magnitude)
- RMS vibration
- Crest factor
- Zero crossing rate

### Multimodal Fusion

Combines vision and accelerometer features for robust detection.

```python
from src.fusion import FusionEngine

engine = FusionEngine(method="rule_based")
result = engine.fuse(vision_features, accel_features)

if result.pothole_detected:
    print(f"Severity: {result.severity} ({result.confidence:.0%})")
```

**Fusion Strategies:**
- **Rule-based**: Configurable thresholds and logic
- **Weighted Average**: Simple weighted combination
- **ML-based**: Trained classifier (optional)

---

## ⚙️ Configuration

All settings are centralized in `config/config.yaml`:

```yaml
vision:
  model_type: "yolov8n"
  confidence_threshold: 0.25
  training:
    epochs: 100
    batch_size: 16

accelerometer:
  window_size_samples: 50
  apply_lowpass_filter: true

fusion:
  method: "rule_based"
  vision_weight: 0.6
  accel_weight: 0.4
```

---

## 📈 Model Training

### Prepare Dataset

```bash
python scripts/prepare_dataset.py --val-split 0.2
```

### Train YOLOv8

```bash
python scripts/train.py --model yolov8n --epochs 100 --batch 16
```

### Train Severity Classifier

The severity classifier can be trained on synthetic data (demo) or real labeled data:

```python
from src.accelerometer import SeverityClassifier

classifier = SeverityClassifier()
classifier.train_synthetic(n_samples_per_class=500)
classifier.save("models/weights/severity_classifier.pkl")
```

---

## 🔬 Accelerometer Data Format

Expected CSV format for sensor data:

```csv
timestamp,latitude,longitude,speed,accelerometerX,accelerometerY,accelerometerZ
1492638964.5,40.447444,-79.944188,0.0,0.016998,-0.962234,0.203887
```

Pothole labels format:

```csv
timestamp
1492639065.7
1492639090.8
```

---

## 📊 Severity Levels

| Severity | Peak (g) | RMS (g) | Description |
|----------|----------|---------|-------------|
| None     | < 0.25   | < 0.10  | Normal road |
| Low      | 0.25-0.5 | 0.10-0.15 | Minor bump |
| Medium   | 0.5-1.5  | 0.15-0.5 | Moderate pothole |
| High     | > 1.5    | > 0.5   | Severe pothole |

---

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
black src/ --line-length 88
flake8 src/
```

---

## 📚 API Reference

### PotholeDetector

```python
class PotholeDetector:
    def __init__(self, model_path, confidence_threshold=0.25)
    def detect(self, image) -> List[Detection]
    def detect_batch(self, images) -> List[List[Detection]]
    def visualize(self, image, detections) -> np.ndarray
```

### FusionEngine

```python
class FusionEngine:
    def __init__(self, method="rule_based", vision_weight=0.6)
    def fuse(self, vision_features, accel_features) -> FusionResult
```

### AlertManager

```python
class AlertManager:
    def __init__(self, debounce_seconds=2.0)
    def process(self, result) -> Optional[Alert]
    def add_callback(self, callback)
```

---

## 🔮 Future Enhancements

- [ ] Real-time camera integration
- [ ] GPS-based heatmap generation
- [ ] Cloud synchronization for fleet management
- [ ] Mobile app integration
- [ ] Edge deployment (Raspberry Pi, Jetson)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines first.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
