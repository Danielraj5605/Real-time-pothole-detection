# Multimodal Pothole Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Architecture-Layered-brightgreen.svg" alt="Architecture">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

A production-grade **offline multimodal ML pipeline** for pothole detection and severity classification. Combines computer vision (YOLOv8) with accelerometer signal processing using a **modular event-driven layered architecture**.

---

## 🎯 Features

- **✅ Modular Event-Driven Layered Architecture** - Clean separation of concerns
- **✅ Vision Pipeline** - YOLOv8-based pothole detection
- **✅ Accelerometer Pipeline** - Signal processing with severity classification
- **✅ Multimodal Fusion** - Combines vision + accelerometer for robust detection
- **✅ JSON Configuration** - Environment-specific settings
- **✅ Dependency Injection** - Testable and extensible
- **✅ Event-Driven** - Async event bus for loose coupling
- **✅ Persistence Layer** - SQLite database for pothole and alert storage
- **✅ Alert System** - Multiple delivery channels (Console, Buzzer, LED)
- **✅ Production Ready** - Comprehensive logging, error handling, and resource management

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

### 2. Run Main Application (Recommended)

```bash
# Run with default (development) configuration
python main.py

# Run with production configuration
set POTHOLE_ENV=production  # Windows
python main.py

# Or on Linux/Mac
export POTHOLE_ENV=production
python main.py
```

### 3. Run Architecture Demo

```bash
# Demonstrates the complete architecture with mock sensors
python demo_architecture.py
```

---

## 📁 Project Structure (New Architecture)

```
Real-time-pothole-detection/
│
├── config/                      # ✅ JSON Configuration (NO YAML)
│   ├── config.json             # Base configuration
│   ├── development.json        # Dev overrides
│   ├── production.json         # Prod overrides
│   └── testing.json            # Test overrides
│
├── src/
│   ├── domain/                 # ✅ Pure Business Logic (No Dependencies)
│   │   ├── entities/           # Core entities (Pothole, Alert, SensorData)
│   │   └── services/           # Domain services (Fusion, Severity, Proximity)
│   │
│   ├── infrastructure/         # ✅ External Integrations
│   │   ├── sensors/            # Camera, Accelerometer, GPS adapters
│   │   ├── ml/                 # YOLOv8 detector adapter
│   │   ├── persistence/        # SQLite repository
│   │   └── alerts/             # Alert channels (Console, Buzzer, LED)
│   │
│   ├── application/            # ✅ Use Cases & Orchestration
│   │   ├── config/             # Configuration & DI container
│   │   ├── events/             # Event bus & domain events
│   │   └── services/           # Application services
│   │
│   ├── vision/                 # Legacy vision code (wrapped by adapters)
│   ├── accelerometer/          # Legacy accelerometer code
│   └── fusion/                 # ⚠️ DEPRECATED (use new architecture)
│
├── main.py                     # ✅ Main application entry point
├── demo_architecture.py        # ✅ Architecture demonstration
├── demo_legacy.py              # Legacy demo (for reference)
│
├── ARCHITECTURE.md             # Architecture documentation
├── FINAL_REPORT.md             # Implementation report
├── CODEBASE_AUDIT.md           # Cleanup and audit report
└── README.md                   # This file
```

---

## 🏗️ Architecture Overview

### Layered Architecture

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

### Design Principles

- **Dependency Inversion** - All dependencies point inward
- **Separation of Concerns** - Each layer has a single responsibility
- **Event-Driven** - Loose coupling via event bus
- **Dependency Injection** - Components wired via JSON configuration
- **Testability** - Mock implementations for all external dependencies

---

## 📊 Usage Examples

### Basic Detection Pipeline

```python
import asyncio
from src.application.config import get_container

async def main():
    # Get dependency container
    container = get_container()
    
    # Get services (all dependencies auto-wired)
    detection_service = container.get_detection_service()
    
    # Process a frame
    pothole = await detection_service.process_frame()
    
    if pothole:
        print(f"Pothole detected: {pothole.severity.value}")
        print(f"Location: ({pothole.latitude}, {pothole.longitude})")
        print(f"Confidence: {pothole.confidence:.2f}")

asyncio.run(main())
```

### Alert Management

```python
from src.application.config import get_container

container = get_container()
alert_service = container.get_alert_service()

# Add custom alert callback
def my_alert_handler(alert):
    print(f"🚨 {alert.message}")

alert_service.add_callback(my_alert_handler)

# Check proximity (in your detection loop)
await alert_service.check_proximity(
    current_lat=40.4474,
    current_lon=-79.9442,
    known_potholes=detected_potholes
)

# Get alert history
history = alert_service.get_history(severity='WARNING', limit=10)

# Get statistics
stats = alert_service.get_statistics()
print(f"Total alerts: {stats['total_alerts']}")
```

### Configuration

Edit `config/config.json`:

```json
{
  "hardware": {
    "mode": "mock"  // Change to "real" for actual hardware
  },
  "detection": {
    "min_confidence": 0.5,
    "accel_threshold": 1.5,
    "frame_rate": 15
  },
  "alerts": {
    "enabled": true,
    "channels": {
      "console": {"enabled": true},
      "buzzer": {"enabled": false},
      "led": {"enabled": false}
    }
  }
}
```

---

## ⚙️ Configuration Management

All configuration is **JSON-based** (no YAML in new architecture):

- `config/config.json` - Base configuration
- `config/development.json` - Development overrides
- `config/production.json` - Production overrides
- `config/testing.json` - Testing overrides

Set environment via `POTHOLE_ENV` variable:

```bash
# Windows
set POTHOLE_ENV=production

# Linux/Mac
export POTHOLE_ENV=production
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

---

## 🧪 Testing

The architecture supports easy testing with mock implementations:

```python
# All components have mock versions for testing
from src.infrastructure.sensors.adapters import MockCamera, MockAccelerometer, MockGPS
from src.infrastructure.ml.adapters import MockDetector

# No hardware required for development!
```

---

## 📊 Database

Potholes and alerts are automatically saved to SQLite:

- **Location**: `data/database/potholes.db`
- **Tables**: `potholes`, `alerts`
- **Features**: Geographic queries, indexing, foreign keys

---

## 🔮 Future Enhancements

- [ ] REST API (FastAPI)
- [ ] WebSocket server for real-time updates
- [ ] CLI interface
- [ ] Web dashboard
- [ ] Cloud storage integration
- [ ] Multi-user support
- [ ] Authentication system
- [ ] Mobile app integration

---

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete architecture documentation
- **[FINAL_REPORT.md](FINAL_REPORT.md)** - Implementation report with test results
- **[CODEBASE_AUDIT.md](CODEBASE_AUDIT.md)** - Code cleanup and audit report
- **[implementation_plan.md](implementation_plan.md)** - Original implementation plan
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage guide

---

## ⚠️ Migration from Legacy Code

If you're using the old architecture (`src/fusion/`, `demo.py`):

1. **Legacy code is deprecated** - Use new architecture instead
2. **Configuration**: Migrate from `config.yaml` to `config/config.json`
3. **Entry point**: Use `main.py` instead of `demo.py`
4. **Imports**: Use new architecture modules:
   - ❌ `from src.fusion import FusionEngine`
   - ✅ `from src.application.config import get_container`

See **[CODEBASE_AUDIT.md](CODEBASE_AUDIT.md)** for migration details.

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

---

## ✅ Status

**Implementation: COMPLETE ✅**
- Architecture: 100% compliant
- Tests: All passing
- Documentation: Complete
- Production ready: Yes

See **[FINAL_REPORT.md](FINAL_REPORT.md)** for detailed status.
