# 🔍 Complete Architectural Audit: Real-time Pothole Detection System
## Part 4: Legacy Modules & Execution Flow

---

## ⚠️ Legacy Modules (Preserved for Compatibility)

### `/src/vision/` - Legacy Vision Module

**Status:** ⚠️ Wrapped by new architecture but still functional

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `detector.py` | 381 | PotholeDetector class - YOLOv8 inference | Wrapped by `YOLOv8Detector` adapter |
| `features.py` | - | VisionFeatures - bbox area, aspect ratio | Still used |
| `trainer.py` | - | VisionTrainer - YOLOv8 training utilities | Used by `scripts/train.py` |

#### `detector.py` - Legacy Detector

**Key class:** `PotholeDetector`

**Methods:**
- `detect(source)` - Run detection on image/array
- `detect_batch(sources)` - Batch inference
- `visualize(image, detections)` - Draw bounding boxes
- `get_best_detection(detections)` - Get highest confidence

**Returns:** List of `Detection` objects with:
- `bbox` - Bounding box coordinates
- `confidence` - Detection confidence
- `class_id`, `class_name` - Classification
- Helper methods: `center()`, `area()`, `width()`, `height()`, `aspect_ratio()`

**Migration note:** New code should use `infrastructure/ml/adapters/yolov8_detector.py` instead.

---

### `/src/accelerometer/` - Legacy Accelerometer Module

**Status:** ⚠️ Still used for signal processing

| File | Lines | Purpose |
|------|-------|---------|
| `processor.py` | 399 | AccelerometerProcessor - CSV loading, windowing, filtering |
| `classifier.py` | - | SeverityClassifier - ML-based severity prediction |
| `features.py` | - | AccelFeatures - peak, RMS, zero-crossing extraction |

#### `processor.py` - Signal Processing

**Key class:** `AccelerometerProcessor`

**Features:**
- CSV/streaming data loading
- Sliding window extraction
- Digital filtering (lowpass, highpass)
- Baseline removal
- Magnitude calculation

**Methods:**
- `load_csv(filepath)` - Load accelerometer CSV
- `process_file(filepath)` - Process and yield windows
- `process_array(x, y, z)` - Process numpy arrays
- `find_pothole_windows(windows, timestamps)` - Label windows

**Returns:** `AccelWindow` objects with:
- `accel_x`, `accel_y`, `accel_z` - 3-axis acceleration
- `magnitude` - 3D magnitude
- `timestamps` - Time array
- `latitude`, `longitude` - GPS coordinates (if available)

**Configuration:**
- `window_size` - Samples per window (default 50)
- `overlap_ratio` - Window overlap (default 0.5)
- `sample_rate` - Sample rate in Hz (default 50)
- `filter_cutoff` - Lowpass cutoff frequency (default 10 Hz)

---

### `/src/fusion/` - DEPRECATED Fusion Engine

**Status:** ⚠️⚠️ DEPRECATED - Use `domain/services/fusion_service.py` instead

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `engine.py` | 437 | Old FusionEngine class | **DEPRECATED** |
| `rules.py` | - | Fusion rules | **DEPRECATED** |
| `alerts.py` | - | Old alert system | **DEPRECATED** |

#### Why deprecated?

The old fusion engine was monolithic and tightly coupled. The new architecture provides:
- ✅ Cleaner separation of concerns
- ✅ Better testability
- ✅ Simpler configuration
- ✅ Event-driven design

**Migration path:**
```python
# Old (deprecated)
from src.fusion import FusionEngine
engine = FusionEngine(method='rule_based')
result = engine.fuse(vision_features, accel_features)

# New (recommended)
from src.application.config import get_container
container = get_container()
service = container.get_detection_service()
pothole = await service.process_frame()
```

---

## 🔄 Complete Execution Flow

### Flow 1: Mock Demo Mode (`python main.py`)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Application Startup                                      │
└─────────────────────────────────────────────────────────────┘
    │
    ├─► Parse CLI arguments
    ├─► setup_logging() → logs/pothole_detection.log
    ├─► load_config() → config/config.json
    │
    └─► run_mock_demo() [async function]
        │
        ┌─────────────────────────────────────────────────────┐
        │ 2. Dependency Injection                             │
        └─────────────────────────────────────────────────────┘
        │
        ├─► get_container() → DependencyContainer (singleton)
        │   ├─► Load config/config.json
        │   └─► Initialize empty _instances dict
        │
        ├─► container.get_event_bus()
        │   ├─► Create EventBus
        │   └─► asyncio.create_task(event_bus.start())
        │
        ├─► container.get_detection_service()
        │   ├─► get_camera() → MockCamera (hardware.mode='mock')
        │   │   ├─► camera.initialize()
        │   │   └─► camera.calibrate()
        │   │
        │   ├─► get_accelerometer() → MockAccelerometer
        │   │   ├─► accel.initialize()
        │   │   └─► accel.calibrate()
        │   │
        │   ├─► get_gps() → MockGPS
        │   │   ├─► gps.initialize()
        │   │   └─► gps.calibrate()
        │   │
        │   ├─► get_detector() → MockDetector
        │   │   └─► detector.initialize()
        │   │
        │   ├─► get_fusion_service() → FusionService
        │   │   └─► Configure weights from config
        │   │
        │   └─► Create DetectionService with all dependencies
        │
        ├─► container.get_alert_service()
        ├─► container.get_reporting_service()
        └─► container.get_repository() → SQLiteRepository
            └─► Create database tables if not exist
        
        ┌─────────────────────────────────────────────────────┐
        │ 3. Detection Loop (10 cycles)                       │
        └─────────────────────────────────────────────────────┘
        │
        └─► FOR i in range(10):
            │
            └─► await detection_service.process_frame()
                │
                ├─► 📷 camera.capture_frame()
                │   └─► MockCamera returns synthetic frame
                │
                ├─► 📊 accelerometer.read()
                │   └─► MockAccelerometer returns random accel data
                │
                ├─► 🌍 gps.read()
                │   └─► MockGPS returns coordinates with jitter
                │
                ├─► 🤖 detector.detect(frame)
                │   └─► MockDetector returns random detections
                │
                ├─► 🔀 fusion_service.fuse(detections, accel_data)
                │   ├─► Extract best detection confidence
                │   ├─► Calculate acceleration magnitude
                │   ├─► Normalize scores (0-1)
                │   ├─► fusion_score = vision_weight * vision_score
                │   │                 + accel_weight * accel_score
                │   └─► Decision: detected if fusion_score >= threshold
                │
                └─► IF pothole detected:
                    │
                    ├─► Create Pothole entity
                    │   ├─► Assign GPS coordinates
                    │   ├─► Classify severity (Severity.from_metrics)
                    │   │   └─► Based on accel_peak, confidence, bbox_area
                    │   └─► Generate UUID
                    │
                    ├─► 📡 event_bus.publish(PotholeDetectedEvent)
                    │   └─► Event queued for async processing
                    │
                    ├─► 💾 repository.save_pothole(pothole)
                    │   └─► INSERT INTO potholes table
                    │
                    └─► Return pothole
        
        ┌─────────────────────────────────────────────────────┐
        │ 4. Report Generation                                │
        └─────────────────────────────────────────────────────┘
        │
        ├─► reporting_service.generate_summary(detected_potholes)
        │   ├─► Count total detections
        │   ├─► Group by severity
        │   └─► Return statistics dict
        │
        └─► Log summary to console
        
        ┌─────────────────────────────────────────────────────┐
        │ 5. Cleanup                                          │
        └─────────────────────────────────────────────────────┘
        │
        ├─► event_bus.stop()
        ├─► container.cleanup()
        │   ├─► camera.cleanup()
        │   ├─► accelerometer.cleanup()
        │   ├─► gps.cleanup()
        │   ├─► detector.cleanup()
        │   └─► repository.cleanup()
        │
        └─► Exit
```

---

### Flow 2: Live Camera Detection (`python live_detection.py`)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Initialization                                           │
└─────────────────────────────────────────────────────────────┘
    │
    ├─► Parse CLI arguments (--camera, --model, --confidence, etc.)
    ├─► setup_logging() → logs/live_detection.log
    ├─► load_config() → config/config.json
    │
    ├─► Create LiveDetectionConfig
    │   └─► Merge config.json + CLI args
    │
    ├─► Create YOLOv8Detector
    │   ├─► detector = YOLOv8Detector(model_path, confidence, iou)
    │   └─► detector.initialize()
    │       ├─► Load YOLO model from .pt file
    │       ├─► Move to GPU/CPU
    │       └─► Get class names
    │
    └─► Create LiveDetectionService
        ├─► service = LiveDetectionService(detector, config, callback)
        └─► service.initialize()
            ├─► Open camera (cv2.VideoCapture)
            ├─► Configure resolution, FPS
            ├─► Capture warmup frames
            └─► Verify detector is loaded

┌─────────────────────────────────────────────────────────────┐
│ 2. Main Detection Loop                                      │
└─────────────────────────────────────────────────────────────┘
    │
    └─► service.run() [WHILE running]:
        │
        ├─► 📷 capture_frame()
        │   ├─► ret, frame = cap.read()
        │   └─► Return BGR numpy array
        │
        ├─► 🤖 detect_on_frame(frame)
        │   ├─► detector.detect(frame)
        │   │   ├─► YOLO inference
        │   │   │   └─► model(frame, conf=threshold, iou=threshold)
        │   │   ├─► Extract boxes, confidences, classes
        │   │   └─► Return List[DetectionResult]
        │   │
        │   └─► Convert to detection dicts
        │
        ├─► IF detections found:
        │   ├─► Call on_detection callback
        │   │   ├─► Calculate severity estimate
        │   │   │   └─► Based on confidence + bbox area
        │   │   └─► Log detection with severity
        │   │
        │   └─► IF save_detections:
        │       └─► _save_frame(frame, detections)
        │           └─► Save to results/live_detections/
        │
        ├─► 🎨 visualize_detections(frame, detections, fps)
        │   ├─► FOR each detection:
        │   │   ├─► Draw bounding box (green)
        │   │   ├─► Draw confidence label
        │   │   └─► Draw class name
        │   │
        │   ├─► Draw FPS counter (top-left)
        │   ├─► Draw detection count
        │   └─► Return annotated frame
        │
        ├─► 🖥️ cv2.imshow(window_name, frame)
        │   └─► Display frame in window
        │
        ├─► ⏱️ Calculate FPS
        │   ├─► Track frame timestamps
        │   └─► Average over last 30 frames
        │
        └─► ⌨️ Handle keyboard input
            ├─► key = cv2.waitKey(1)
            ├─► IF key == 'q': stop()
            └─► IF key == 's': _save_frame()

┌─────────────────────────────────────────────────────────────┐
│ 3. Shutdown                                                  │
└─────────────────────────────────────────────────────────────┘
    │
    └─► service.stop()
        ├─► cv2.destroyAllWindows()
        ├─► cap.release()
        ├─► detector.cleanup()
        └─► Log statistics
```

---

### Flow 3: Data Flow Through Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL WORLD                            │
└─────────────────────────────────────────────────────────────┘
    │
    ├─► Camera (OpenCV)
    ├─► Accelerometer (MPU6050)
    └─► GPS (NEO-6M)
    
    ▼
    
┌─────────────────────────────────────────────────────────────┐
│              INFRASTRUCTURE LAYER (Adapters)                 │
└─────────────────────────────────────────────────────────────┘
    │
    ├─► OpenCVCamera.capture_frame() → Frame
    ├─► MPU6050Accelerometer.read() → AccelData
    └─► NEO6MGPS.read() → GPSData
    
    ▼
    
┌─────────────────────────────────────────────────────────────┐
│              APPLICATION LAYER (Services)                    │
└─────────────────────────────────────────────────────────────┘
    │
    └─► DetectionService.process_frame()
        │
        ├─► YOLOv8Detector.detect(frame) → [DetectionResult]
        │
        └─► Pass to Domain Layer ▼
    
┌─────────────────────────────────────────────────────────────┐
│              DOMAIN LAYER (Business Logic)                   │
└─────────────────────────────────────────────────────────────┘
    │
    └─► FusionService.fuse(detections, accel_data)
        │
        ├─► Calculate fusion score
        ├─► Apply business rules
        └─► Return FusionResult
        
        ▼
        
    └─► Create Pothole entity
        │
        └─► Severity.from_metrics(accel, conf, bbox)
            └─► Apply severity classification rules
    
    ▼
    
┌─────────────────────────────────────────────────────────────┐
│              APPLICATION LAYER (Events)                      │
└─────────────────────────────────────────────────────────────┘
    │
    └─► EventBus.publish(PotholeDetectedEvent)
        │
        └─► Dispatch to subscribers ▼
    
┌─────────────────────────────────────────────────────────────┐
│              INFRASTRUCTURE LAYER (Persistence)              │
└─────────────────────────────────────────────────────────────┘
    │
    ├─► SQLiteRepository.save_pothole()
    │   └─► INSERT INTO potholes
    │
    └─► AlertService.check_proximity()
        │
        └─► ConsoleAlert.send_alert()
            └─► Print to console
```

---

**Continue to Part 5 for key files, navigation guide, and takeaways...**
