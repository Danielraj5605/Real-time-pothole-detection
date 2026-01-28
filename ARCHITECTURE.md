# Pothole Detection System - Layered Architecture

## Architecture Overview

This project implements a **Modular Event-Driven Layered Architecture** designed for maintainability, testability, and future scalability.

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│              (CLI, REST API, WebSocket - Future)             │
├─────────────────────────────────────────────────────────────┤
│                    APPLICATION LAYER                         │
│         (Services, Use Cases, Event Bus, DTOs)               │
├─────────────────────────────────────────────────────────────┤
│                      DOMAIN LAYER                            │
│        (Entities, Value Objects, Domain Services)            │
├─────────────────────────────────────────────────────────────┤
│                  INFRASTRUCTURE LAYER                        │
│    (Sensors, ML Models, Persistence, External APIs)          │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles Applied

| Principle | Implementation |
|-----------|----------------|
| **DRY** (Don't Repeat Yourself) | Shared sensor interfaces, common data models |
| **SRP** (Single Responsibility) | Each module has one clear purpose |
| **SoC** (Separation of Concerns) | Clear layer boundaries |
| **OCP** (Open/Closed) | Plugin-based extensibility |
| **DIP** (Dependency Inversion) | Interfaces for all external dependencies |
| **EDA** (Event-Driven) | Asynchronous event bus for loose coupling |

## Directory Structure

```
src/
├── domain/                      # Core business logic (no dependencies)
│   ├── entities/               # Domain entities
│   │   ├── pothole.py         # Pothole entity with business rules
│   │   ├── sensor_data.py     # Sensor data models
│   │   └── alert.py           # Alert entity
│   └── services/              # Domain services
│       ├── fusion_service.py  # Multimodal fusion logic
│       ├── severity_classifier.py
│       └── proximity_calculator.py
│
├── infrastructure/             # External integrations
│   ├── sensors/
│   │   ├── interfaces/        # Abstract sensor interfaces
│   │   │   ├── sensor_interface.py
│   │   │   ├── camera_interface.py
│   │   │   ├── accelerometer_interface.py
│   │   │   └── gps_interface.py
│   │   └── adapters/          # Concrete implementations
│   │       ├── opencv_camera.py
│   │       ├── mpu6050_accelerometer.py
│   │       ├── neo6m_gps.py
│   │       └── mock_sensors.py  # For testing
│   └── ml/
│       ├── interfaces/
│       │   └── detector_interface.py
│       └── adapters/
│           ├── yolov8_detector.py
│           └── mock_detector.py
│
├── application/                # Use cases and orchestration
│   ├── services/
│   │   ├── detection_service.py  # Main detection pipeline
│   │   ├── alert_service.py      # Alert management
│   │   └── reporting_service.py  # Statistics and reports
│   └── events/
│       ├── event_bus.py          # Central event dispatcher
│       ├── pothole_detected.py   # Domain events
│       └── alert_triggered.py
│
└── presentation/               # APIs and UIs (Future)
    ├── api/                    # REST API
    ├── cli/                    # Command-line interface
    └── websocket/              # Real-time updates
```

## Key Components

### 1. Domain Layer

**Pure business logic with zero external dependencies.**

- **Pothole Entity**: Core domain model with severity classification
- **Sensor Data**: Accelerometer, GPS data models
- **Fusion Service**: Combines vision + accelerometer data
- **Severity Classifier**: Business rules for severity levels
- **Proximity Calculator**: Distance calculations and alert triggers

### 2. Infrastructure Layer

**All external integrations wrapped in adapters.**

#### Sensor Interfaces (Ports)
- `SensorInterface`: Base interface for all sensors
- `CameraInterface`: Camera abstraction
- `AccelerometerInterface`: Accelerometer abstraction
- `GPSInterface`: GPS abstraction

#### Sensor Adapters
- `OpenCVCamera`: OpenCV camera implementation
- `MPU6050Accelerometer`: MPU6050 I2C accelerometer
- `NEO6MGPS`: NEO-6M GPS with NMEA parsing
- `MockSensors`: Testing without hardware

#### ML Adapters
- `YOLOv8Detector`: Wraps YOLOv8 model
- `MockDetector`: Testing without ML model

### 3. Application Layer

**Orchestrates domain logic and infrastructure.**

#### Services
- **DetectionService**: Main pipeline orchestrator
  - Captures sensor data
  - Runs ML detection
  - Applies fusion logic
  - Publishes events
  
- **AlertService**: Proximity-based alerts
  - Checks for nearby potholes
  - Generates alerts
  - Sends through multiple channels
  
- **ReportingService**: Statistics and analytics
  - Summary statistics
  - Time series data
  - Geographic distribution

#### Event Bus
- Asynchronous event dispatcher
- Observer pattern implementation
- Loose coupling between components

## Usage Example

```python
import asyncio
from src.infrastructure.sensors.adapters import MockCamera, MockAccelerometer, MockGPS
from src.infrastructure.ml.adapters import MockDetector
from src.domain.services.fusion_service import FusionService
from src.application.services.detection_service import DetectionService, DetectionConfig
from src.application.events.event_bus import EventBus

async def main():
    # 1. Initialize infrastructure (adapters)
    camera = MockCamera()
    accelerometer = MockAccelerometer()
    gps = MockGPS()
    detector = MockDetector()
    
    # Initialize all
    camera.initialize()
    accelerometer.initialize()
    gps.initialize()
    detector.initialize()
    
    # 2. Create domain services
    fusion_service = FusionService(
        vision_weight=0.6,
        accel_weight=0.4
    )
    
    # 3. Create application services
    event_bus = EventBus()
    config = DetectionConfig(
        min_confidence=0.5,
        accel_threshold=1.5
    )
    
    detection_service = DetectionService(
        camera=camera,
        accelerometer=accelerometer,
        gps=gps,
        detector=detector,
        fusion_service=fusion_service,
        event_bus=event_bus,
        config=config
    )
    
    # 4. Subscribe to events
    async def on_pothole_detected(event):
        print(f"Pothole detected: {event.pothole.severity.value}")
    
    event_bus.subscribe(PotholeDetectedEvent, on_pothole_detected)
    
    # 5. Start event bus
    bus_task = asyncio.create_task(event_bus.start())
    
    # 6. Process frames
    for _ in range(10):
        pothole = await detection_service.process_frame()
        if pothole:
            print(f"Detected: {pothole.to_dict()}")
        await asyncio.sleep(0.1)
    
    # 7. Cleanup
    event_bus.stop()
    await bus_task
    camera.cleanup()
    accelerometer.cleanup()
    gps.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Benefits

### Maintainability
- Clear separation of concerns
- Each module has single responsibility
- Easy to locate and fix bugs

### Testability
- Dependency injection allows mocking
- Mock sensors for testing without hardware
- Mock detector for testing without ML model
- Pure domain logic can be unit tested

### Extensibility
- Add new sensors by implementing interfaces
- Add new alert channels without modifying core
- Add new issue types (future civic platform)
- Plugin-based architecture

### Scalability
- Event-driven design allows async processing
- Can distribute to multiple services later
- Infrastructure can be swapped (Pi → Cloud)

## Future Extensions

### Civic Infrastructure Platform

The architecture is designed to easily extend to a broader civic reporting platform:

```python
# Future: Abstract issue type
class CivicIssue(ABC):
    @abstractmethod
    def determine_severity(self) -> IssueSeverity:
        pass

# Current: Pothole is a CivicIssue
class PotholeIssue(CivicIssue):
    def determine_severity(self) -> IssueSeverity:
        # Existing logic
        pass

# Future: New issue types
class StreetlightIssue(CivicIssue):
    def determine_severity(self) -> IssueSeverity:
        # New logic
        pass
```

### Planned Features
- REST API for remote access
- WebSocket for real-time updates
- Mobile app integration
- Cloud storage (Firebase/PostgreSQL)
- Multi-user support with authentication
- Voting and verification system

## Testing

```bash
# Run with mock sensors (no hardware needed)
python examples/demo_architecture.py

# Run with real sensors (on Raspberry Pi)
python examples/demo_real_hardware.py
```

## Migration from Old Code

The existing code in `src/accelerometer`, `src/vision`, and `src/fusion` is preserved and wrapped by adapters in the infrastructure layer. This allows:

1. **Gradual migration**: Old code still works
2. **No breaking changes**: Existing scripts continue to function
3. **New architecture benefits**: New code uses clean architecture
4. **Easy testing**: Mock implementations for development

## References

- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Event-Driven Architecture](https://martinfowler.com/articles/201701-event-driven.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
