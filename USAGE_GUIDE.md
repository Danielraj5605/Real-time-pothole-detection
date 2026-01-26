# 📖 Complete Usage Guide - Multimodal Pothole Detection System

## 🎯 System Overview

This system detects potholes using **TWO independent methods** that work together:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT DATA                                   │
├──────────────────────────┬──────────────────────────────────────┤
│  📷 Road Images          │  📊 Accelerometer CSV                │
│  (from camera)           │  (from phone/sensor)                 │
└────────┬─────────────────┴──────────────┬───────────────────────┘
         │                                │
         ▼                                ▼
┌─────────────────────┐          ┌──────────────────────┐
│  VISION PIPELINE    │          │  ACCEL PIPELINE      │
│                     │          │                      │
│  YOLOv8 Detection   │          │  Signal Processing   │
│  ↓                  │          │  ↓                   │
│  Features:          │          │  Features:           │
│  • Confidence       │          │  • Peak (g)          │
│  • Bbox size        │          │  • RMS vibration     │
│  • Location         │          │  • Crest factor      │
└────────┬────────────┘          └──────────┬───────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
              ┌──────────────────┐
              │  FUSION ENGINE   │
              │                  │
              │  Combines both   │
              │  for final       │
              │  decision        │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  FINAL OUTPUT    │
              │                  │
              │  • Detected: Yes │
              │  • Severity: High│
              │  • Confidence: 85%│
              │  • GPS Location  │
              └──────────────────┘
```

---

## 📊 Understanding the Output

### **1. Vision Pipeline Output**

When processing an image, you'll see:

```
📷 VISION PIPELINE DEMO
============================================================

Loading image: road_image.jpg
Image size: 640x360

🔍 Running pothole detection...

📊 Detection Results:
   Potholes found: 2

   Detection 1:
   - Confidence: 85.3%         ← How sure the AI is (0-100%)
   - Bounding Box: (120, 200, 180, 280)  ← [x1, y1, x2, y2] pixels
   - Area: 4800 pixels²        ← Size of detected region
   - Aspect Ratio: 1.33        ← Width/Height ratio

   Detection 2:
   - Confidence: 72.1%
   - Bounding Box: (300, 150, 350, 220)
   - Area: 3500 pixels²
   - Aspect Ratio: 1.40

📈 Extracted Features:
   - Detected: True
   - Best Confidence: 85.3%
   - Normalized Area: 0.0370   ← Relative to image size
   - Num Detections: 2
   - Visual Severity Hint: medium
```

**What Each Metric Means:**

| Metric | Range | Meaning | Example |
|--------|-------|---------|---------|
| **Confidence** | 0-100% | AI certainty | 85% = Very confident |
| **Bounding Box** | Pixel coords | Location in image | (x1, y1, x2, y2) |
| **Area** | Pixels² | Size of detection | 4800px² = medium |
| **Normalized Area** | 0-1 | % of image covered | 0.05 = 5% of image |
| **Aspect Ratio** | Number | Shape (W/H) | 1.0 = square, 2.0 = wide |

**Severity Hints from Vision:**
- `low`: Small detection (< 3% of image)
- `medium`: Medium detection (3-10% of image)
- `high`: Large detection (> 10% of image) with high confidence

---

### **2. Accelerometer Pipeline Output**

Processes sensor data in **1-second sliding windows**:

```
📊 ACCELEROMETER PIPELINE DEMO
============================================================

📂 Processing: trip1_sensors.csv
   Generated 87 analysis windows

🔬 Sample Window Analysis:

   Window 15 (t=1492639034.5s):
   - Peak: 1.85g               ← Maximum impact force
   - RMS: 0.52g                ← Average vibration
   - Crest Factor: 3.56        ← Impact sharpness
   - Severity: high (89%)      ← ML classification
   - Location: (40.4528, -79.9461)  ← GPS coordinates
```

**Understanding Accelerometer Values:**

| Metric | What It Measures | Pothole Indication |
|--------|------------------|-------------------|
| **Peak (g)** | Maximum acceleration spike | > 1.5g = Severe impact |
| **RMS (g)** | Root Mean Square vibration | > 0.5g = High roughness |
| **Crest Factor** | Peak ÷ RMS | > 3.0 = Sharp sudden impact |
| **Zero Crossing** | Signal frequency | Higher = more vibration |

**Severity Classification:**

```
┌─────────────┬──────────────┬──────────────┬─────────────────────┐
│  Severity   │   Peak (g)   │   RMS (g)    │   Description       │
├─────────────┼──────────────┼──────────────┼─────────────────────┤
│  None       │   < 0.25     │   < 0.10     │  Smooth road        │
│  Low        │  0.25 - 0.5  │  0.10 - 0.15 │  Minor bump         │
│  Medium     │  0.5 - 1.5   │  0.15 - 0.5  │  Moderate pothole   │
│  High       │   > 1.5      │   > 0.5      │  Severe pothole     │
└─────────────┴──────────────┴──────────────┴─────────────────────┘
```

**Real-World Examples:**
- **0.2g** - Normal driving on good road
- **0.5g** - Small bump or crack
- **1.0g** - Medium pothole
- **2.0g** - Large pothole (noticeable jolt)
- **3.0g+** - Severe pothole (potential damage)

---

### **3. Fusion Engine Output**

Combines both pipelines for the final decision:

```
🔀 MULTIMODAL FUSION DEMO
============================================================

🔄 Fusing 15 candidate windows...

   Fusion 1:
   - FusionResult(DETECTED, severity=high, conf=0.87, vision=0.85, peak=1.85g)
     │            │          │            │          │           │
     │            │          │            │          │           └─ Accel peak value
     │            │          │            │          └─ Vision confidence
     │            │          │            └─ Combined confidence (0-1)
     │            │          └─ Final severity classification
     │            └─ Detection status (DETECTED or clear)
     └─ Result object type

   🚨 NEW ALERT: Alert #1: HIGH pothole @ (40.4528, -79.9461) (conf=87%)
```

**Fusion Logic Explained:**

```python
# Scenario 1: Both pipelines agree
IF vision_detected AND accel_peak > 0.3g:
    → DETECTED with HIGH confidence
    → confidence = min(1.0, (0.6 × vision_conf + 0.4 × accel_score) × 1.2)
    → Confidence BOOSTED by 20% for agreement

# Scenario 2: Only vision detects
ELIF vision_detected AND accel_peak < 0.3g:
    → Check combined score
    → combined = 0.6 × vision_conf + 0.4 × accel_score
    → DETECTED if combined > 0.5

# Scenario 3: Only accelerometer detects
ELIF accel_peak > 0.3g AND NOT vision_detected:
    → Check combined score
    → combined = 0.6 × vision_conf + 0.4 × accel_score
    → DETECTED if combined > 0.5

# Scenario 4: Neither detects
ELSE:
    → CLEAR (no pothole)
```

**Confidence Interpretation:**
- **0.0 - 0.3**: Low confidence (probably not a pothole)
- **0.3 - 0.5**: Uncertain (borderline case)
- **0.5 - 0.7**: Moderate confidence (likely a pothole)
- **0.7 - 0.9**: High confidence (definitely a pothole)
- **0.9 - 1.0**: Very high confidence (both pipelines strongly agree)

---

## 🧪 How to Test and Verify

### **Step 1: Quick Test (Demo)**

Run the complete demo to see all pipelines in action:

```bash
python demo.py
```

**Expected Output:**
- ✅ Vision detections with bounding boxes
- ✅ Accelerometer analysis with severity
- ✅ Fusion results combining both
- ✅ Visualization saved to `results/demo_outputs/vision_detection.jpg`
- ✅ Logs saved to `logs/pothole_detection.log`

---

### **Step 2: Test Vision Pipeline Only**

Create a test script `test_vision.py`:

```python
from src.vision import PotholeDetector, VisionFeatureExtractor
import cv2

# Initialize detector
detector = PotholeDetector("yolov8n.pt", confidence_threshold=0.25)
extractor = VisionFeatureExtractor()

# Test on an image
image_path = "Datasets/images/20250216_164325.jpg"
detections = detector.detect(image_path)

print(f"\n{'='*60}")
print(f"VISION TEST RESULTS")
print(f"{'='*60}")
print(f"Image: {image_path}")
print(f"Detections found: {len(detections)}\n")

for i, det in enumerate(detections, 1):
    print(f"Detection {i}:")
    print(f"  Confidence: {det.confidence:.2%}")
    print(f"  Bounding box: {det.bbox}")
    print(f"  Area: {det.area:.0f} pixels²")
    print(f"  Center: {det.center}")
    print()

# Extract features
img = cv2.imread(image_path)
h, w = img.shape[:2]
features = extractor.extract(detections, w, h)

print(f"Extracted Features:")
print(f"  Detected: {features.detected}")
print(f"  Confidence: {features.confidence:.2%}")
print(f"  Normalized area: {features.bbox_area_normalized:.4f}")
print(f"  Severity hint: {extractor.compute_severity_hint(features)}")

# Save visualization
vis = detector.visualize(image_path, detections)
cv2.imwrite("test_vision_output.jpg", vis)
print(f"\n✅ Visualization saved: test_vision_output.jpg")
```

Run it:
```bash
python test_vision.py
```

---

### **Step 3: Test Accelerometer Pipeline Only**

Create `test_accel.py`:

```python
from src.accelerometer import (
    AccelerometerProcessor, 
    AccelFeatureExtractor, 
    SeverityClassifier
)

# Initialize
processor = AccelerometerProcessor(
    window_size=50,      # 1 second at 50Hz
    overlap_ratio=0.5,   # 50% overlap
    apply_filter=True    # Apply lowpass filter
)
extractor = AccelFeatureExtractor()
classifier = SeverityClassifier()

# Train classifier on synthetic data
print("Training severity classifier...")
metrics = classifier.train_synthetic(n_samples_per_class=500)
print(f"✅ Training complete: {metrics['test_accuracy']:.2%} accuracy\n")

# Process sensor file
csv_path = "Datasets/Pothole/trip1_sensors.csv"
print(f"Processing: {csv_path}\n")

pothole_count = 0
for i, window in enumerate(processor.process_file(csv_path)):
    features = extractor.extract(window)
    
    # Check if it's a pothole candidate
    if features.peak_acceleration > 0.3:  # Threshold
        prediction = classifier.predict(features)
        
        print(f"Window {i+1} (t={window.start_time:.1f}s):")
        print(f"  Peak: {features.peak_acceleration:.3f}g")
        print(f"  RMS: {features.rms_vibration:.3f}g")
        print(f"  Crest: {features.crest_factor:.2f}")
        print(f"  Severity: {prediction.severity} ({prediction.confidence:.0%})")
        
        if window.latitude and window.longitude:
            print(f"  GPS: ({window.latitude:.4f}, {window.longitude:.4f})")
        print()
        
        pothole_count += 1
        
        if pothole_count >= 10:  # Limit output
            break

print(f"✅ Found {pothole_count} pothole candidates")
```

Run it:
```bash
python test_accel.py
```

---

### **Step 4: Test Complete Fusion**

Create `test_fusion.py`:

```python
from src.vision import PotholeDetector, VisionFeatureExtractor
from src.accelerometer import (
    AccelerometerProcessor, 
    AccelFeatureExtractor, 
    SeverityClassifier
)
from src.fusion import FusionEngine, AlertManager

# Initialize all components
print("Initializing pipelines...")
detector = PotholeDetector("yolov8n.pt")
vision_extractor = VisionFeatureExtractor()

processor = AccelerometerProcessor()
accel_extractor = AccelFeatureExtractor()
classifier = SeverityClassifier()
classifier.train_synthetic(n_samples_per_class=200)

fusion = FusionEngine(method="rule_based", vision_weight=0.6, accel_weight=0.4)
alerts = AlertManager(debounce_seconds=2.0, min_severity='low')

# Alert callback
def on_alert(alert):
    severity_emoji = {'low': '⚠️', 'medium': '🔶', 'high': '🔴'}
    emoji = severity_emoji.get(alert.severity, '❓')
    print(f"\n{emoji} ALERT #{alert.id}: {alert.severity.upper()} pothole")
    print(f"   Confidence: {alert.confidence:.0%}")
    print(f"   Location: ({alert.latitude}, {alert.longitude})")
    print(f"   Vision: {alert.vision_confidence:.0%}, Accel: {alert.accel_peak:.2f}g")

alerts.add_callback(on_alert)

# Process vision
print("\n1️⃣ Processing image...")
image_path = "Datasets/images/20250216_164325.jpg"
detections = detector.detect(image_path)

import cv2
img = cv2.imread(image_path)
h, w = img.shape[:2]
vision_features = vision_extractor.extract(detections, w, h)

print(f"   Vision: {len(detections)} detections, conf={vision_features.confidence:.0%}")

# Process accelerometer
print("\n2️⃣ Processing accelerometer data...")
csv_path = "Datasets/Pothole/trip1_sensors.csv"
windows = list(processor.process_file(csv_path))[:20]  # First 20 windows

print(f"   Accel: {len(windows)} windows processed")

# Fusion
print("\n3️⃣ Running fusion...")
fusion_count = 0

for window in windows:
    accel_features = accel_extractor.extract(window)
    
    # Only fuse if accelerometer detects something
    if accel_features.peak_acceleration > 0.25:
        prediction = classifier.predict(accel_features)
        
        result = fusion.fuse(
            vision_features if fusion_count == 0 else None,  # Use vision once
            accel_features,
            latitude=window.latitude,
            longitude=window.longitude,
            accel_severity=prediction
        )
        
        print(f"\n   Fusion {fusion_count + 1}: {result}")
        
        # Process alert
        alert = alerts.process(result)
        
        fusion_count += 1
        if fusion_count >= 5:  # Limit
            break

# Statistics
print("\n" + "="*60)
stats = alerts.get_statistics()
print(f"SUMMARY:")
print(f"  Total alerts: {stats['total_alerts']}")
print(f"  By severity: {stats['severity_counts']}")
print(f"  Avg confidence: {stats['avg_confidence']:.0%}")
```

Run it:
```bash
python test_fusion.py
```

---

### **Step 5: Check Output Files**

After running the demo or tests, verify these files were created:

```bash
# Visualization
results/demo_outputs/vision_detection.jpg  ← Image with bounding boxes

# Logs
logs/pothole_detection.log                 ← Detailed system logs
logs/pothole_events.db                     ← SQLite database of all detections

# Models (after training)
models/weights/pothole_yolov8n_best.pt     ← Trained YOLO weights
models/weights/severity_classifier.pkl      ← Trained severity classifier
```

**View the SQLite database:**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('logs/pothole_events.db')
df = pd.read_sql_query("SELECT * FROM pothole_events ORDER BY timestamp DESC LIMIT 10", conn)
print(df)
conn.close()
```

---

## 🎓 Training Your Own Model

The demo uses a **pretrained YOLOv8** model (trained on COCO dataset), which won't detect potholes accurately. To train on your pothole images:

### **Step 1: Prepare Dataset**
```bash
python scripts/prepare_dataset.py --val-split 0.2
```

This creates:
- `Datasets/train/` - 989 training images + labels
- `Datasets/val/` - 247 validation images + labels
- `Datasets/pothole_dataset.yaml` - Configuration file

### **Step 2: Train YOLOv8**
```bash
# Quick training (for testing)
python scripts/train.py --epochs 10 --batch 8

# Full training (recommended)
python scripts/train.py --epochs 100 --batch 16 --model yolov8n
```

**Training parameters:**
- `--model`: Model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
  - `n` (nano): Fastest, least accurate
  - `s` (small): Good balance
  - `m` (medium): Better accuracy
  - `l/x`: Best accuracy, slower
- `--epochs`: Training iterations (100-200 recommended)
- `--batch`: Batch size (8-32 depending on GPU memory)

### **Step 3: Use Trained Model**
```python
from src.vision import PotholeDetector

# Use your trained model
detector = PotholeDetector("models/weights/pothole_yolov8n_best.pt")
detections = detector.detect("test_image.jpg")
```

---

## 📊 Expected Performance

### **Vision Pipeline:**
- **Speed**: ~30-100 FPS (depending on model size and hardware)
- **Accuracy**: 85-95% (after training on labeled data)
- **False positives**: Shadows, cracks, road markings

### **Accelerometer Pipeline:**
- **Speed**: Real-time (processes 50 samples/sec)
- **Accuracy**: 90-95% (with trained classifier)
- **False positives**: Speed bumps, railroad crossings

### **Fusion:**
- **Accuracy**: 95-98% (combining both reduces false positives)
- **Confidence**: Higher when both pipelines agree

---

## 🐛 Troubleshooting

### **Issue: No detections found**
```
✅ Check confidence threshold (try lowering to 0.1)
✅ Verify image quality (not too dark/blurry)
✅ Train model on your specific pothole images
```

### **Issue: Too many false positives**
```
✅ Increase confidence threshold (try 0.5-0.7)
✅ Adjust fusion weights (increase vision_weight)
✅ Enable stricter fusion rules
```

### **Issue: Accelerometer not detecting**
```
✅ Check CSV format (columns: timestamp, accelerometerX/Y/Z)
✅ Verify sample rate (default: 50Hz)
✅ Lower peak threshold (try 0.2g instead of 0.3g)
```

---

## 📞 Next Steps

1. **Run the demo**: `python demo.py`
2. **Train the model**: `python scripts/train.py`
3. **Test individual components**: Use the test scripts above
4. **Integrate into your application**: Import the modules you need
5. **Deploy**: Package for edge devices (Raspberry Pi, Jetson, etc.)

For questions or issues, check the logs in `logs/pothole_detection.log`!
