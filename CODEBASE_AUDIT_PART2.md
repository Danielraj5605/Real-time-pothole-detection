# 🔍 Complete Architectural Audit: Real-time Pothole Detection System
## Part 2: Folder Structure & Directory Explanations

---

## 📁 Complete Folder Tree

```
Real-time-pothole-detection/
│
├── 📄 main.py                      # ✅ Main entry point (mock demo & live mode)
├── 📄 live_detection.py            # ✅ Standalone live camera detection script
├── 📄 demo_architecture.py         # Architecture demonstration script
├── 📄 yolov8n.pt                   # Pre-trained YOLOv8 nano model weights (6.5MB)
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # Project documentation
├── 📄 ARCHITECTURE.md              # Architecture documentation
├── 📄 USAGE_GUIDE.md               # Detailed usage guide
├── 📄 implementation_plan.md       # Original implementation plan
├── 📄 .gitignore                   # Git ignore rules
├── 📄 .gitattributes               # Git LFS configuration
│
├── 📁 config/                      # ⚙️ JSON Configuration (Environment-specific)
│   ├── 📄 config.json              # Base configuration (303 lines, comprehensive)
│   ├── 📄 development.json         # Development environment overrides
│   ├── 📄 production.json          # Production environment overrides
│   └── 📄 testing.json             # Testing environment overrides
│
├── 📁 src/                         # 🔧 Source Code (New Architecture)
│   ├── 📄 __init__.py              # Package init
│   │
│   ├── 📁 domain/                  # 🔵 DOMAIN LAYER (Pure Business Logic)
│   │   ├── 📄 __init__.py
│   │   ├── 📁 entities/            # Core domain models
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 pothole.py       # Pothole entity with severity classification
│   │   │   ├── 📄 alert.py         # Alert entity with alert levels
│   │   │   └── 📄 sensor_data.py   # Sensor data models
│   │   └── 📁 services/            # Domain services (pure logic)
│   │       ├── 📄 __init__.py
│   │       ├── 📄 fusion_service.py        # Multimodal fusion logic
│   │       ├── 📄 severity_classifier.py   # Severity classification rules
│   │       └── 📄 proximity_calculator.py  # Distance/Haversine calculations
│   │
│   ├── 📁 application/             # 🟢 APPLICATION LAYER (Orchestration)
│   │   ├── 📄 __init__.py
│   │   ├── 📁 config/              # Configuration & DI
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 dependency_injection.py  # DI Container (304 lines - CRITICAL!)
│   │   │   └── 📄 settings.py              # Config loading utilities
│   │   ├── 📁 events/              # Event-driven system
│   │   │   ├── 📄 __init__.py
│   │   │   ├── 📄 event_bus.py             # Central event dispatcher (124 lines)
│   │   │   ├── 📄 base_event.py            # Base event class
│   │   │   ├── 📄 pothole_detected.py      # PotholeDetectedEvent
│   │   │   └── 📄 alert_triggered.py       # AlertTriggeredEvent
│   │   └── 📁 services/            # Application services
│   │       ├── 📄 __init__.py
│   │       ├── 📄 detection_service.py      # Main detection pipeline (184 lines)
│   │       ├── 📄 live_detection_service.py # Real-time camera service (564 lines)
│   │       ├── 📄 alert_service.py          # Alert generation & delivery (260 lines)
│   │       └── 📄 reporting_service.py      # Statistics & reports
│   │
│   ├── 📁 infrastructure/          # 🟠 INFRASTRUCTURE LAYER (External Integrations)
│   │   ├── 📄 __init__.py
│   │   ├── 📁 sensors/             # Hardware/sensor abstractions
│   │   │   ├── 📁 interfaces/      # Port contracts
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   ├── 📄 sensor_interface.py       # Base sensor interface
│   │   │   │   ├── 📄 camera_interface.py       # Camera abstraction
│   │   │   │   ├── 📄 accelerometer_interface.py # Accelerometer abstraction
│   │   │   │   └── 📄 gps_interface.py          # GPS abstraction
│   │   │   └── 📁 adapters/        # Concrete implementations
│   │   │       ├── 📄 __init__.py
│   │   │       ├── 📄 opencv_camera.py          # OpenCV camera (563 lines)
│   │   │       ├── 📄 mpu6050_accelerometer.py  # MPU6050 I2C accelerometer
│   │   │       ├── 📄 neo6m_gps.py              # NEO-6M GPS with NMEA
│   │   │       └── 📄 mock_sensors.py           # Mock sensors for testing
│   │   ├── 📁 ml/                  # Machine learning adapters
│   │   │   ├── 📁 interfaces/
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   └── 📄 detector_interface.py     # Detector contract
│   │   │   └── 📁 adapters/
│   │   │       ├── 📄 __init__.py
│   │   │       ├── 📄 yolov8_detector.py        # YOLOv8 wrapper (143 lines)
│   │   │       └── 📄 mock_detector.py          # Mock detector for testing
│   │   ├── 📁 persistence/         # Data storage
│   │   │   ├── 📁 interfaces/
│   │   │   │   ├── 📄 __init__.py
│   │   │   │   └── 📄 repository_interface.py   # Repository contract
│   │   │   └── 📁 adapters/
│   │   │       ├── 📄 __init__.py
│   │   │       └── 📄 sqlite_repository.py      # SQLite implementation (274 lines)
│   │   └── 📁 alerts/              # Alert delivery channels
│   │       ├── 📁 interfaces/
│   │       │   ├── 📄 __init__.py
│   │       │   └── 📄 alert_channel_interface.py
│   │       └── 📁 adapters/
│   │           ├── 📄 __init__.py
│   │           ├── 📄 console_alert.py          # Console/log alerts
│   │           ├── 📄 buzzer_alert.py           # Hardware buzzer (Pi)
│   │           └── 📄 led_alert.py              # Hardware LED (Pi)
│   │
│   ├── 📁 vision/                  # ⚠️ LEGACY: Vision processing code
│   │   ├── 📄 __init__.py
│   │   ├── 📄 detector.py          # PotholeDetector class (381 lines)
│   │   ├── 📄 features.py          # Vision feature extraction
│   │   └── 📄 trainer.py           # Model training utilities
│   │
│   ├── 📁 accelerometer/           # ⚠️ LEGACY: Accelerometer processing
│   │   ├── 📄 __init__.py
│   │   ├── 📄 processor.py         # AccelerometerProcessor (399 lines)
│   │   ├── 📄 classifier.py        # Severity classifier (ML-based)
│   │   └── 📄 features.py          # Accelerometer feature extraction
│   │
│   ├── 📁 fusion/                  # ⚠️ DEPRECATED: Old fusion engine
│   │   ├── 📄 __init__.py
│   │   ├── 📄 engine.py            # FusionEngine (437 lines) - DEPRECATED
│   │   ├── 📄 rules.py             # Fusion rules - DEPRECATED
│   │   └── 📄 alerts.py            # Old alert system - DEPRECATED
│   │
│   └── 📁 utils/                   # Utilities
│       ├── 📄 __init__.py
│       ├── 📄 config_loader.py     # Config loading helpers
│       └── 📄 logger.py            # Logging configuration
│
├── 📁 scripts/                     # 🔨 Training & Utility Scripts
│   ├── 📄 train.py                 # YOLOv8 training script (138 lines)
│   ├── 📄 prepare_dataset.py       # Dataset preparation (254 lines)
│   └── 📄 evaluate.py              # Model evaluation
│
├── 📁 Datasets/                    # 📊 Training Data (~2000+ images)
│   ├── 📁 Pothole/                 # Accelerometer CSV data (5 trips)
│   │   ├── 📄 trip1_sensors.csv
│   │   ├── 📄 trip1_potholes.csv
│   │   ├── 📄 trip2_sensors.csv
│   │   ├── 📄 trip2_potholes.csv
│   │   ├── 📄 trip3_sensors.csv
│   │   ├── 📄 trip3_potholes.csv
│   │   ├── 📄 trip4_sensors.csv
│   │   ├── 📄 trip4_potholes.csv
│   │   ├── 📄 trip5_sensors.csv
│   │   └── 📄 trip5_potholes.csv
│   ├── 📁 Pothole_Image_Data/      # Raw pothole images (106+ images)
│   ├── 📁 images/                  # Processed images
│   ├── 📁 labels/                  # YOLO format labels (2009 files)
│   ├── 📁 train/                   # Training split
│   │   ├── 📁 images/              # 592 training images
│   │   └── 📁 labels/              # 592 training labels
│   ├── 📁 val/                     # Validation split
│   │   ├── 📁 images/              # 221 validation images
│   │   └── 📁 labels/              # 221 validation labels
│   └── 📄 pothole_dataset.yaml     # YOLO dataset configuration
│
├── 📁 data/                        # 💾 Runtime Data
│   └── 📁 database/
│       └── 📄 potholes.db          # SQLite database (potholes, alerts)
│
├── 📁 logs/                        # 📝 Log Files
│   ├── 📄 pothole_detection.log
│   └── 📄 live_detection.log
│
├── 📁 models/                      # 🤖 Model Storage (weights, checkpoints)
│   ├── 📁 weights/                 # Trained model weights
│   └── 📁 yolo_training/           # Training outputs
│
└── 📁 results/                     # 📸 Detection outputs, visualizations
    └── 📁 live_detections/         # Saved detection frames
```

---

## 📂 Directory Explanations

### `/config/` - Configuration Management

**Purpose:** Centralized JSON-based configuration with environment-specific overrides

| File | Lines | Purpose |
|------|-------|---------|
| `config.json` | 303 | **Master configuration** - ALL configurable parameters |
| `development.json` | 229 | Development overrides (mock hardware mode) |
| `production.json` | 345 | Production overrides (real hardware, optimized) |
| `testing.json` | 360 | Testing overrides (reduced thresholds) |

**How it fits:** Environment is selected via `POTHOLE_ENV` environment variable. The `DependencyContainer` loads the appropriate config at startup.

**Key sections in config.json:**
- `hardware` - Camera settings, hardware mode (mock/real)
- `vision` - YOLO model path, training params, inference thresholds
- `accelerometer` - Sample rate, windowing, filtering
- `gps` - GPS settings, simulation mode
- `fusion` - Fusion method, weights, thresholds
- `alerts` - Alert channels, distance thresholds
- `persistence` - Database path
- `live_detection` - Camera FPS, visualization settings

---

### `/src/domain/` - Pure Business Logic Layer 🔵

**Purpose:** Core business logic with ZERO external dependencies

**Why it exists:** Isolates business rules from technical concerns. Can be tested without any infrastructure.

#### `/domain/entities/` - Domain Models

| File | Lines | Purpose |
|------|-------|---------|
| `pothole.py` | 109 | **Core domain entity** - Pothole dataclass with Severity enum |
| `alert.py` | - | Alert entity with AlertLevel enum |
| `sensor_data.py` | - | Data classes for sensor readings |

**Key classes:**
- `Pothole` - id, lat/lon, severity, confidence, accel_peak, bbox_area, detected_at
- `Severity` enum - LOW/MEDIUM/HIGH with `from_metrics()` business rule
- `Alert` - id, pothole_id, level, message, distance, created_at

**Business rules:**
- `Severity.from_metrics(accel_peak, confidence, bbox_area)` - Determines severity
- `Pothole.distance_to(lat, lon)` - Haversine distance calculation
- `Pothole.should_alert_at_distance(distance_m)` - Alert threshold logic

#### `/domain/services/` - Domain Services

| File | Lines | Purpose |
|------|-------|---------|
| `fusion_service.py` | 106 | **Multimodal fusion logic** - Combines vision + accelerometer |
| `severity_classifier.py` | - | Rule-based severity classification |
| `proximity_calculator.py` | - | Haversine distance calculations |

**Key logic:**
- `FusionService.fuse()` - Weighted fusion of vision confidence + accel magnitude
- Returns `FusionResult` with detection decision and metrics

---

**Continue to Part 3 for Application & Infrastructure layers...**
