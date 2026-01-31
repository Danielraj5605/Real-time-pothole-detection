# 🔍 Complete Architectural Audit: Real-time Pothole Detection System
## Part 1: Overview & Architecture

**Generated:** 2026-02-01  
**Project Version:** 1.0.0  
**Status:** Production Ready ✅

---

## 📋 Project Overview

**Project Name:** Multimodal Pothole Detection System  
**Primary Language:** Python 3.8+  
**Architecture:** Modular Event-Driven Layered Architecture  
**ML Framework:** YOLOv8 (Ultralytics)  
**Database:** SQLite  

### What This Project Does

This is a **production-grade offline multimodal ML pipeline** for real-time pothole detection and severity classification. It combines:

- **Computer Vision** (YOLOv8 object detection) for visual pothole identification
- **Accelerometer Signal Processing** for vehicle vibration detection
- **Sensor Fusion** to combine both modalities for robust detection
- **GPS Tracking** for pothole geolocation
- **Alert System** for proximity-based warnings

### Key Features

- ✅ Modular Event-Driven Layered Architecture - Clean separation of concerns
- ✅ Vision Pipeline - YOLOv8-based pothole detection
- ✅ Accelerometer Pipeline - Signal processing with severity classification
- ✅ Multimodal Fusion - Combines vision + accelerometer for robust detection
- ✅ JSON Configuration - Environment-specific settings
- ✅ Dependency Injection - Testable and extensible
- ✅ Event-Driven - Async event bus for loose coupling
- ✅ Persistence Layer - SQLite database for pothole and alert storage
- ✅ Alert System - Multiple delivery channels (Console, Buzzer, LED)
- ✅ Production Ready - Comprehensive logging, error handling, and resource management

---

## 🏗️ Architecture Summary

The system follows a **Clean/Layered Architecture** pattern with 4 main layers:

```
┌─────────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER (Future)                     │
│                  REST API, WebSocket, CLI                    │
├─────────────────────────────────────────────────────────────┤
│              APPLICATION LAYER                               │
│   DetectionService, AlertService, ReportingService          │
│   EventBus, Configuration, DependencyInjection              │
├─────────────────────────────────────────────────────────────┤
│              DOMAIN LAYER                                    │
│   Pothole, Alert, FusionService, SeverityClassifier         │
│   ProximityCalculator (PURE - No Dependencies)              │
├─────────────────────────────────────────────────────────────┤
│              INFRASTRUCTURE LAYER                            │
│   Sensors, ML, Persistence, Alerts                          │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles Applied

| Principle | Implementation |
|-----------|----------------|
| **DRY** (Don't Repeat Yourself) | Shared sensor interfaces, common data models |
| **SRP** (Single Responsibility) | Each module has one clear purpose |
| **SoC** (Separation of Concerns) | Clear layer boundaries |
| **OCP** (Open/Closed) | Plugin-based extensibility |
| **DIP** (Dependency Inversion) | Interfaces for all external dependencies |
| **EDA** (Event-Driven) | Asynchronous event bus for loose coupling |

### Key Design Patterns Used

| Pattern | Implementation |
|---------|----------------|
| **Dependency Injection** | `DependencyContainer` manages all component creation |
| **Adapter Pattern** | Sensors/ML wrapped with interfaces for swappability |
| **Observer Pattern** | `EventBus` for loose coupling between components |
| **Repository Pattern** | `SQLiteRepository` abstracts data persistence |
| **Port/Adapter (Hexagonal)** | Interfaces define contracts, adapters implement them |
| **Factory Pattern** | Container creates instances based on configuration |

---

## 📊 Project Statistics

- **Total Source Files:** 65+ files
- **Total Lines of Code:** ~15,000+ lines
- **Configuration Files:** 4 JSON files
- **Dataset Images:** 2,009 labeled images
- **Training/Validation Split:** 592 train / 221 validation
- **Accelerometer Data:** 5 trip datasets with sensor readings
- **Database Tables:** 2 (potholes, alerts)

---

## 🎯 Entry Points

### 1. `main.py` - Primary Entry Point

**Purpose:** Main application entry point with two modes

**Modes:**
- **Mock Demo Mode** (default): Tests architecture with simulated sensors
- **Live Mode** (`--live`): Runs real-time camera detection

**Usage:**
```bash
# Mock demo
python main.py

# Live camera detection
python main.py --live

# Live with custom settings
python main.py --live --camera 1 --confidence 0.5
```

### 2. `live_detection.py` - Live Camera Entry Point

**Purpose:** Standalone real-time pothole detection from camera feed

**Features:**
- Real-time YOLOv8 inference
- FPS monitoring and display
- Bounding box visualization
- Frame saving capability
- Keyboard controls (q=quit, s=save)

**Usage:**
```bash
# Run with default settings
python live_detection.py

# Use specific camera
python live_detection.py --camera 1

# Custom model and confidence
python live_detection.py --model best.pt --confidence 0.5

# Headless mode (no display)
python live_detection.py --no-display

# Save detection frames
python live_detection.py --save
```

### 3. `demo_architecture.py` - Architecture Demo

**Purpose:** Demonstrates the complete architecture with mock sensors

---

## 🔄 System Flow Overview

### High-Level Detection Flow

```
1. Sensor Data Collection
   ├── Camera captures frame
   ├── Accelerometer reads vibration data
   └── GPS reads current coordinates

2. ML Detection
   └── YOLOv8 processes frame → detections with confidence scores

3. Multimodal Fusion
   ├── Combine vision confidence + accelerometer magnitude
   └── Apply weighted fusion with configurable thresholds

4. Pothole Entity Creation
   ├── Assign GPS coordinates
   ├── Classify severity (LOW/MEDIUM/HIGH)
   └── Generate unique ID

5. Event Publishing
   └── Publish PotholeDetectedEvent to EventBus

6. Event Handling
   ├── Save to SQLite database
   ├── Check proximity for alerts
   └── Trigger custom callbacks

7. Alert Delivery
   ├── Console alerts
   ├── Hardware alerts (Buzzer/LED)
   └── Event notifications
```

---

## 📦 Dependencies

### Core ML & Deep Learning
- `torch` ≥2.0.0 - PyTorch for deep learning
- `torchvision` ≥0.15.0 - Vision utilities
- `ultralytics` ≥8.0.0 - YOLOv8 framework

### Scientific Computing
- `numpy` ≥1.21.0 - Array operations
- `scipy` ≥1.9.0 - Signal processing (filtering)
- `pandas` ≥1.4.0 - CSV data handling

### Machine Learning
- `scikit-learn` ≥1.0.0 - ML utilities and classifiers

### Image Processing
- `opencv-python` ≥4.7.0 - Camera capture & visualization
- `Pillow` ≥9.0.0 - Image processing

### Configuration & Utilities
- `PyYAML` ≥6.0 - YAML config (for YOLO datasets)
- `python-dotenv` ≥1.0.0 - Environment variables
- `tqdm` ≥4.64.0 - Progress bars

---

## 🗄️ Database Schema

### SQLite Database: `data/database/potholes.db`

#### Table: `potholes`
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (PK) | UUID identifier |
| `latitude` | REAL | GPS latitude |
| `longitude` | REAL | GPS longitude |
| `severity` | TEXT | LOW/MEDIUM/HIGH |
| `confidence` | REAL | ML detection confidence (0-1) |
| `accel_peak` | REAL | Peak acceleration (g-force) |
| `bbox_area` | INTEGER | Bounding box area (pixels) |
| `image_path` | TEXT | Path to saved image (optional) |
| `detected_at` | TEXT | ISO timestamp |
| `is_verified` | INTEGER | Human verification flag (0/1) |

**Indexes:**
- `idx_potholes_location` on (latitude, longitude)
- `idx_potholes_detected_at` on (detected_at)

#### Table: `alerts`
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (PK) | UUID identifier |
| `pothole_id` | TEXT (FK) | Reference to pothole |
| `level` | TEXT | INFO/WARNING/DANGER |
| `message` | TEXT | Alert message |
| `distance_meters` | REAL | Distance to pothole |
| `created_at` | TEXT | ISO timestamp |
| `acknowledged` | INTEGER | Acknowledgment flag (0/1) |

**Indexes:**
- `idx_alerts_pothole_id` on (pothole_id)

---

## 🔌 Hardware Support

### Supported Hardware

| Component | Model | Interface | Status |
|-----------|-------|-----------|--------|
| Camera | Any OpenCV-compatible webcam | USB/CSI | ✅ Supported |
| Camera | Raspberry Pi Camera Module | CSI | ✅ Supported |
| Accelerometer | MPU6050 | I2C | ✅ Supported |
| GPS | NEO-6M | Serial/UART | ✅ Supported |
| Alert Output | GPIO Buzzer | GPIO | ✅ Supported (Pi) |
| Alert Output | GPIO LED | GPIO | ✅ Supported (Pi) |

### Mock Hardware

All hardware components have **mock implementations** for testing without physical devices:
- `MockCamera` - Generates synthetic frames
- `MockAccelerometer` - Simulates vibration data
- `MockGPS` - Simulates GPS coordinates with jitter
- `MockDetector` - Returns random detections

---

**Continue to Part 2 for detailed folder structure and file explanations...**
