# 🔍 Complete Architectural Audit: Real-time Pothole Detection System
## Part 3: Application & Infrastructure Layers

---

## 📂 Application Layer 🟢

**Purpose:** Orchestrates domain logic with infrastructure. Coordinates use cases and manages system flow.

### `/application/config/` - Configuration & Dependency Injection

| File | Lines | Purpose | Importance |
|------|-------|---------|------------|
| `dependency_injection.py` | 304 | **DI Container** - Creates all system components | ⭐⭐⭐⭐⭐ CRITICAL |
| `settings.py` | - | Config loading and environment detection | ⭐⭐⭐ |

#### `dependency_injection.py` - THE GLUE OF THE SYSTEM

**What it does:** Factory for all system components. Manages lifecycle and dependencies.

**Key class:** `DependencyContainer`

**Key methods:**

| Method | Returns | Purpose |
|--------|---------|---------|
| `get_camera()` | `CameraInterface` | Creates OpenCV or Mock camera based on config |
| `get_accelerometer()` | `AccelerometerInterface` | Creates real or mock accelerometer |
| `get_gps()` | `GPSInterface` | Creates real or mock GPS |
| `get_detector()` | `DetectorInterface` | Creates YOLOv8 or mock detector |
| `get_fusion_service()` | `FusionService` | Creates fusion service with weights |
| `get_detection_service()` | `DetectionService` | Creates main detection pipeline |
| `get_alert_service()` | `AlertService` | Creates alert management service |
| `get_repository()` | `SQLiteRepository` | Creates database repository |
| `get_event_bus()` | `EventBus` | Creates event dispatcher |
| `cleanup()` | None | Releases all resources |

**Usage pattern:**
```python
from src.application.config import get_container
container = get_container()  # Singleton
service = container.get_detection_service()
```

**Why critical:** This file wires the entire system together. Breaking it breaks everything.

---

### `/application/events/` - Event-Driven System

| File | Lines | Purpose |
|------|-------|---------|
| `event_bus.py` | 124 | **Central event dispatcher** - Observer pattern |
| `base_event.py` | - | Base event class |
| `pothole_detected.py` | - | PotholeDetectedEvent |
| `alert_triggered.py` | - | AlertTriggeredEvent |

#### `event_bus.py` - Event Dispatcher

**What it does:** Implements Observer pattern for loose coupling between components.

**Key methods:**
- `subscribe(event_type, handler)` - Register event handler
- `publish(event)` - Publish event to queue
- `start()` - Start async event processing loop
- `stop()` - Stop event bus

**Usage:**
```python
# Subscribe
event_bus.subscribe(PotholeDetectedEvent, my_handler)

# Publish
await event_bus.publish(PotholeDetectedEvent(pothole=pothole))
```

**Why it exists:** Decouples components. Services don't need to know about each other.

---

### `/application/services/` - Application Services

| File | Lines | Purpose | Key Responsibility |
|------|-------|---------|-------------------|
| `detection_service.py` | 184 | **Main detection pipeline** | Orchestrates sensors → ML → fusion → events |
| `live_detection_service.py` | 564 | **Real-time camera service** | Camera loop, FPS, visualization |
| `alert_service.py` | 260 | **Alert management** | Proximity checks, alert delivery |
| `reporting_service.py` | - | **Statistics & reports** | Summary generation |

#### `detection_service.py` - Main Pipeline Orchestrator

**What it does:** Coordinates the complete detection pipeline

**Flow:**
1. Capture frame from camera
2. Read accelerometer data
3. Read GPS coordinates
4. Run ML detection (YOLOv8)
5. Apply multimodal fusion
6. Create Pothole entity with severity
7. Publish PotholeDetectedEvent

**Defines Ports (Protocols):**
- `CameraPort` - camera interface
- `AccelerometerPort` - accelerometer interface
- `GPSPort` - GPS interface
- `DetectorPort` - ML detector interface

**Key method:**
```python
async def process_frame() -> Optional[Pothole]:
    # Returns Pothole if detected, None otherwise
```

#### `live_detection_service.py` - Real-Time Camera Service

**What it does:** Comprehensive real-time camera detection with visualization

**Features:**
- Camera initialization with warmup
- Frame capture and preprocessing
- YOLO inference with configurable thresholds
- Bounding box visualization
- FPS calculation and display
- Frame saving with timestamps
- Keyboard controls (q=quit, s=save)

**Key methods:**
- `initialize()` - Setup camera and detector
- `capture_frame()` - Get single frame
- `detect_on_frame(frame)` - Run YOLO inference
- `visualize_detections(frame, detections, fps)` - Draw boxes and info
- `run()` - Main detection loop
- `stop()` - Cleanup resources

**Configuration:** Uses `LiveDetectionConfig` dataclass with 25+ parameters

#### `alert_service.py` - Alert Management

**What it does:** Generates and delivers proximity-based alerts

**Features:**
- Proximity checking using Haversine distance
- Alert history management (deque)
- Callback system for custom handlers
- Statistics tracking
- Multiple delivery channels

**Key methods:**
- `check_proximity(lat, lon, known_potholes)` - Check for nearby potholes
- `add_callback(callback)` - Register alert handler
- `get_history(severity, limit)` - Get alert history
- `get_statistics()` - Get alert stats

---

## 📂 Infrastructure Layer 🟠

**Purpose:** All external integrations wrapped in adapters implementing interfaces.

### `/infrastructure/sensors/` - Hardware Abstractions

#### Interfaces (Ports)

| File | Purpose |
|------|---------|
| `sensor_interface.py` | Base sensor interface with `initialize()`, `calibrate()`, `read()`, `cleanup()` |
| `camera_interface.py` | Camera abstraction with `capture_frame()` |
| `accelerometer_interface.py` | Accelerometer abstraction with `read()`, `set_range()` |
| `gps_interface.py` | GPS abstraction with `read()`, `get_coordinates()` |

#### Adapters (Implementations)

| File | Lines | Purpose | Hardware |
|------|-------|---------|----------|
| `opencv_camera.py` | 563 | OpenCV camera wrapper | Webcam, Pi Camera |
| `mpu6050_accelerometer.py` | - | MPU6050 I2C accelerometer | MPU6050 sensor |
| `neo6m_gps.py` | - | NEO-6M GPS with NMEA parsing | NEO-6M GPS module |
| `mock_sensors.py` | - | **Mock implementations** | None (testing) |

#### `opencv_camera.py` - Camera Adapter

**What it does:** Wraps OpenCV VideoCapture with comprehensive configuration

**Features:**
- Configurable resolution, FPS, backend
- Warmup frames for camera stabilization
- Auto-reconnection on failure
- FPS monitoring
- Context manager support

**Key methods:**
- `initialize()` - Open camera and configure
- `calibrate()` - Capture warmup frames
- `capture_frame()` - Get single frame
- `set_resolution(width, height)` - Change resolution
- `set_exposure(value)` - Manual exposure control
- `is_healthy()` - Check camera status

**Supports:**
- Standard USB webcams
- Raspberry Pi Camera Module (via OpenCV backend)
- GStreamer pipelines

#### `mock_sensors.py` - Mock Implementations

**What it does:** Provides mock sensors for testing without hardware

**Classes:**
- `MockCamera` - Returns synthetic frames (solid color or noise)
- `MockAccelerometer` - Simulates vibration with random peaks
- `MockGPS` - Simulates coordinates with jitter

**Why critical:** Enables development and testing without physical hardware.

---

### `/infrastructure/ml/` - Machine Learning Adapters

#### Interfaces

| File | Purpose |
|------|---------|
| `detector_interface.py` | Abstract detector with `detect()`, `initialize()`, `is_loaded()` |

#### Adapters

| File | Lines | Purpose |
|------|-------|---------|
| `yolov8_detector.py` | 143 | **YOLOv8 wrapper** - Main ML adapter |
| `mock_detector.py` | - | Mock detector for testing |

#### `yolov8_detector.py` - YOLOv8 Wrapper

**What it does:** Wraps Ultralytics YOLO model to implement DetectorInterface

**Key methods:**
- `initialize()` - Load YOLO model to GPU/CPU
- `detect(image)` - Run inference, return DetectionResult list
- `is_loaded()` - Check if model is loaded
- `cleanup()` - Release model resources

**Returns:** List of `DetectionResult` with:
- `bbox` - (x1, y1, x2, y2) coordinates
- `confidence` - Detection confidence (0-1)
- `class_id` - Class ID
- `class_name` - Class name string

**Configuration:**
- `model_path` - Path to .pt weights
- `confidence_threshold` - Min confidence (default 0.25)
- `iou_threshold` - NMS threshold (default 0.45)
- `device` - 'cuda', 'cpu', or auto-detect

---

### `/infrastructure/persistence/` - Data Storage

#### Interfaces

| File | Purpose |
|------|---------|
| `repository_interface.py` | Abstract repository with save/get/query methods |

#### Adapters

| File | Lines | Purpose |
|------|-------|---------|
| `sqlite_repository.py` | 274 | **SQLite implementation** |

#### `sqlite_repository.py` - Database Repository

**What it does:** Manages SQLite database for potholes and alerts

**Key methods:**
- `save_pothole(pothole)` - Insert/update pothole
- `get_pothole(id)` - Get by ID
- `get_all_potholes()` - Get all potholes
- `get_potholes_by_location(lat, lon, radius_m)` - Geographic query
- `save_alert(alert)` - Save alert
- `get_alerts_by_pothole(pothole_id)` - Get alerts for pothole

**Database initialization:**
- Auto-creates database file and directory
- Creates tables with indexes
- Sets up foreign keys

**Geographic queries:**
- Uses bounding box for efficiency
- Filters by actual Haversine distance

---

### `/infrastructure/alerts/` - Alert Delivery

#### Interfaces

| File | Purpose |
|------|---------|
| `alert_channel_interface.py` | Abstract alert channel with `send_alert(alert)` |

#### Adapters

| File | Purpose | Platform |
|------|---------|----------|
| `console_alert.py` | Prints alerts to console/log | All |
| `buzzer_alert.py` | GPIO buzzer control | Raspberry Pi |
| `led_alert.py` | GPIO LED control | Raspberry Pi |

**Why multiple channels:** Allows flexible alert delivery. Can enable/disable channels via config.

---

**Continue to Part 4 for Legacy modules and execution flow...**
