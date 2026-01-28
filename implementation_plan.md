# Software Architecture Proposal for Pothole Detection System

## Executive Summary

This document proposes a comprehensive software architecture for your **Real-Time Pothole Detection and Driver Alerting System**, designed with future scalability towards a **Civic Infrastructure Reporting Platform**. The architecture applies proven software design principles and patterns to create a maintainable, extensible, and robust system.

---

## Architecture Principles Applied

Before diving into the architecture, let's understand the key principles that guide our design:

| Principle | Acronym | Description | Application in Your Project |
|-----------|---------|-------------|----------------------------|
| **Don't Repeat Yourself** | DRY | Avoid code/logic duplication | Shared sensor interfaces, common data models |
| **Single Responsibility** | SRP | Each module does one thing well | Separate modules for detection, classification, alerting |
| **Separation of Concerns** | SoC | Divide system into distinct layers | Data acquisition → Processing → Storage → Presentation |
| **Open/Closed Principle** | OCP | Open for extension, closed for modification | Plugin-based issue type handlers |
| **Dependency Inversion** | DIP | Depend on abstractions, not concretions | Interfaces for sensors, storage, notification channels |
| **Event-Driven Architecture** | EDA | React to events asynchronously | Decouple detection from alerting via message queue |

---

## Recommended Architecture: **Modular Event-Driven Layered Architecture**

### Why This Architecture?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ARCHITECTURE SELECTION MATRIX                       │
├─────────────────────────┬─────────────────┬────────────┬───────────────────┤
│ Architecture            │ Current Project │ Scalability│ Civic Platform    │
├─────────────────────────┼─────────────────┼────────────┼───────────────────┤
│ Monolithic              │ ✓ Simple        │ ✗ Poor     │ ✗ Not Suitable    │
│ Microservices           │ ✗ Overkill      │ ✓ Excellent│ ✓ Future Ready    │
│ Layered (N-Tier)        │ ✓ Appropriate   │ ◐ Moderate │ ◐ Needs Extension │
│ Event-Driven + Layered  │ ✓ Perfect Fit   │ ✓ Excellent│ ✓ Future Ready    │
│ Hexagonal/Clean         │ ✓ Good          │ ✓ Good     │ ✓ Good            │
└─────────────────────────┴─────────────────┴────────────┴───────────────────┘

✓ = Well Suited  ◐ = Moderate  ✗ = Poor Fit
```

**Recommended: Modular Event-Driven Layered Architecture** - It provides the right balance between simplicity for current needs and extensibility for future civic platform integration.

---

## High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           POTHOLE DETECTION SYSTEM                                │
│                    Modular Event-Driven Layered Architecture                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                         PRESENTATION LAYER                                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │ │
│  │  │   REST API  │  │  WebSocket  │  │  CLI/TUI    │  │  Mobile App (Future)│ │ │
│  │  │   Server    │  │  Realtime   │  │  Interface  │  │  Flutter/React      │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │ │
│  └────────────────────────────────────────┬────────────────────────────────────┘ │
│                                           │                                       │
│  ┌────────────────────────────────────────┼────────────────────────────────────┐ │
│  │                         APPLICATION LAYER                                    │ │
│  │  ┌─────────────────┐  ┌───────────────┐  ┌───────────────────┐              │ │
│  │  │ Detection       │  │ Alert         │  │ Reporting         │              │ │
│  │  │ Service         │  │ Service       │  │ Service           │              │ │
│  │  └────────┬────────┘  └───────┬───────┘  └────────┬──────────┘              │ │
│  │           │                   │                    │                         │ │
│  │  ┌────────┴───────────────────┴────────────────────┴────────────┐           │ │
│  │  │                    EVENT BUS (Message Queue)                  │           │ │
│  │  │              Local: asyncio.Queue / Redis                     │           │ │
│  │  │              Future: RabbitMQ / Kafka                         │           │ │
│  │  └───────────────────────────────────────────────────────────────┘           │ │
│  └────────────────────────────────────────┬────────────────────────────────────┘ │
│                                           │                                       │
│  ┌────────────────────────────────────────┼────────────────────────────────────┐ │
│  │                          DOMAIN LAYER                                        │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐ │ │
│  │  │   Pothole     │  │   Sensor      │  │   Location    │  │   Alert       │ │ │
│  │  │   Entity      │  │   Reading     │  │   Entity      │  │   Entity      │ │ │
│  │  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘ │ │
│  │                                                                              │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │ │
│  │  │   Fusion      │  │   Severity    │  │   Proximity   │                    │ │
│  │  │   Strategy    │  │   Classifier  │  │   Calculator  │                    │ │
│  │  └───────────────┘  └───────────────┘  └───────────────┘                    │ │
│  └────────────────────────────────────────┬────────────────────────────────────┘ │
│                                           │                                       │
│  ┌────────────────────────────────────────┼────────────────────────────────────┐ │
│  │                      INFRASTRUCTURE LAYER                                    │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐ │ │
│  │  │   Camera      │  │   MPU6050     │  │   GPS         │  │   Buzzer/LED  │ │ │
│  │  │   Adapter     │  │   Adapter     │  │   Adapter     │  │   Adapter     │ │ │
│  │  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘ │ │
│  │                                                                              │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │ │
│  │  │   SQLite      │  │   YOLOv8      │  │   Cloud API   │                    │ │
│  │  │   Repository  │  │   ML Engine   │  │   (Future)    │                    │ │
│  │  └───────────────┘  └───────────────┘  └───────────────┘                    │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Layer Description

### 1. Infrastructure Layer (Adapters & Ports)

This layer contains all external integrations - hardware sensors, databases, ML engines, and external APIs. Each component is wrapped in an **adapter** that implements a **port (interface)**.

```
infrastructure/
├── sensors/
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── camera_interface.py      # Abstract interface for camera
│   │   ├── accelerometer_interface.py
│   │   └── gps_interface.py
│   ├── adapters/
│   │   ├── raspberry_pi_camera.py   # Concrete implementation
│   │   ├── mpu6050_accelerometer.py
│   │   ├── neo6m_gps.py
│   │   └── mock_sensors.py          # For testing without hardware
│   └── factory.py                   # Creates appropriate adapters
├── ml/
│   ├── interfaces/
│   │   └── detector_interface.py
│   ├── adapters/
│   │   ├── yolov8_detector.py
│   │   └── mock_detector.py
│   └── models/                      # Trained model files
├── persistence/
│   ├── interfaces/
│   │   └── repository_interface.py
│   ├── adapters/
│   │   ├── sqlite_repository.py
│   │   ├── json_repository.py
│   │   └── cloud_repository.py      # Future: Firebase/Cloud
│   └── migrations/
├── alerts/
│   ├── interfaces/
│   │   └── alert_channel_interface.py
│   ├── adapters/
│   │   ├── buzzer_alert.py
│   │   ├── led_alert.py
│   │   ├── console_alert.py
│   │   ├── tts_alert.py
│   │   └── push_notification.py     # Future
│   └── factory.py
└── networking/
    ├── websocket_server.py
    └── http_client.py
```

**DRY Applied:** All sensors implement a common `SensorInterface`, eliminating duplicate read/calibrate logic.

**Example Interface:**

```python
# infrastructure/sensors/interfaces/sensor_interface.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class SensorReading(Generic[T]):
    timestamp_ms: int
    data: T
    quality: float  # 0.0 - 1.0

class SensorInterface(ABC, Generic[T]):
    """Base interface for all sensors - DRY principle"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize sensor hardware"""
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate sensor for accurate readings"""
        pass
    
    @abstractmethod
    def read(self) -> SensorReading[T]:
        """Read current sensor data"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if sensor is functioning correctly"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release sensor resources"""
        pass
```

---

### 2. Domain Layer (Business Logic)

This layer contains core business entities and domain logic. It has **no dependencies** on external frameworks or infrastructure - pure Python.

```
domain/
├── entities/
│   ├── __init__.py
│   ├── pothole.py           # Core pothole entity
│   ├── sensor_data.py       # Unified sensor data model
│   ├── location.py          # GPS coordinates with utilities
│   ├── alert.py             # Alert entity
│   └── severity.py          # Severity enum and rules
├── value_objects/
│   ├── bounding_box.py      # Immutable value objects
│   ├── coordinates.py
│   └── confidence_score.py
├── services/
│   ├── fusion_service.py    # Multimodal fusion logic
│   ├── severity_classifier.py
│   └── proximity_calculator.py
├── events/
│   ├── __init__.py
│   ├── base_event.py
│   ├── pothole_detected.py
│   ├── alert_triggered.py
│   └── sensor_data_received.py
└── specifications/
    ├── alert_specification.py   # Business rules for alerting
    └── fusion_specification.py  # Fusion decision rules
```

**Example Domain Entity:**

```python
# domain/entities/pothole.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import uuid

class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    
    @classmethod
    def from_metrics(cls, accel_peak: float, confidence: float, bbox_area: int) -> 'Severity':
        """Business logic for severity classification - Single Responsibility"""
        if accel_peak > 2.5 and confidence > 0.7 and bbox_area > 10000:
            return cls.HIGH
        elif accel_peak > 1.8 and confidence > 0.6 and bbox_area > 5000:
            return cls.MEDIUM
        return cls.LOW

@dataclass
class Pothole:
    """Core domain entity - immutable after creation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    latitude: float = 0.0
    longitude: float = 0.0
    severity: Severity = Severity.LOW
    confidence: float = 0.0
    accel_peak: float = 0.0
    bbox_area: int = 0
    image_path: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)
    is_verified: bool = False
    
    def distance_to(self, lat: float, lon: float) -> float:
        """Calculate distance using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth's radius in meters
        lat1, lon1 = radians(self.latitude), radians(self.longitude)
        lat2, lon2 = radians(lat), radians(lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def should_alert_at_distance(self, distance_m: float) -> bool:
        """Business rule for alerting based on severity and distance"""
        thresholds = {
            Severity.HIGH: 100,
            Severity.MEDIUM: 50,
            Severity.LOW: 20
        }
        return distance_m <= thresholds.get(self.severity, 20)
```

---

### 3. Application Layer (Use Cases & Orchestration)

This layer contains application-specific use cases that orchestrate domain logic and infrastructure. It also hosts the **Event Bus** for decoupled communication.

```
application/
├── services/
│   ├── __init__.py
│   ├── detection_service.py     # Orchestrates detection pipeline
│   ├── alert_service.py         # Manages alert generation
│   ├── reporting_service.py     # Generates reports
│   └── data_collection_service.py
├── use_cases/
│   ├── detect_pothole.py
│   ├── trigger_proximity_alert.py
│   ├── generate_report.py
│   └── calibrate_sensors.py
├── events/
│   ├── event_bus.py             # Central event dispatcher
│   ├── event_handlers/
│   │   ├── pothole_detected_handler.py
│   │   ├── alert_handler.py
│   │   └── persistence_handler.py
│   └── event_types.py
├── dto/
│   ├── detection_result.py      # Data Transfer Objects
│   ├── alert_request.py
│   └── sensor_packet.py
└── config/
    ├── settings.py              # Application configuration
    └── dependency_injection.py  # DI container
```

**Event Bus Implementation (Event-Driven Architecture):**

```python
# application/events/event_bus.py
from typing import Callable, Dict, List, Type
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

@dataclass
class Event:
    """Base event class"""
    timestamp: datetime
    source: str

class EventBus:
    """
    Central event dispatcher for loose coupling.
    Implements Observer pattern for extensibility.
    """
    
    def __init__(self):
        self._handlers: Dict[Type[Event], List[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._logger = logging.getLogger(__name__)
    
    def subscribe(self, event_type: Type[Event], handler: Callable) -> None:
        """Subscribe a handler to an event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        self._logger.debug(f"Handler {handler.__name__} subscribed to {event_type.__name__}")
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers"""
        await self._queue.put(event)
    
    async def start(self) -> None:
        """Start processing events"""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
    
    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to registered handlers"""
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(f"Handler error: {e}")
    
    def stop(self) -> None:
        """Stop the event bus"""
        self._running = False
```

**Detection Service (Orchestration):**

```python
# application/services/detection_service.py
from typing import Protocol
from dataclasses import dataclass
from domain.entities.pothole import Pothole, Severity
from domain.services.fusion_service import FusionService
from application.events.event_bus import EventBus
from application.events.pothole_detected import PotholeDetectedEvent

class CameraPort(Protocol):
    """Port for camera - Dependency Inversion"""
    def capture_frame(self) -> bytes: ...

class AccelerometerPort(Protocol):
    """Port for accelerometer"""
    def read_acceleration(self) -> dict: ...

class GPSPort(Protocol):
    """Port for GPS"""
    def get_coordinates(self) -> tuple[float, float]: ...

class DetectorPort(Protocol):
    """Port for ML detector"""
    def detect(self, frame: bytes) -> list[dict]: ...

@dataclass
class DetectionConfig:
    min_confidence: float = 0.5
    accel_threshold: float = 1.5
    frame_rate: int = 15

class DetectionService:
    """
    Orchestrates the detection pipeline.
    Depends only on abstractions (DIP).
    """
    
    def __init__(
        self,
        camera: CameraPort,
        accelerometer: AccelerometerPort,
        gps: GPSPort,
        detector: DetectorPort,
        fusion_service: FusionService,
        event_bus: EventBus,
        config: DetectionConfig
    ):
        self._camera = camera
        self._accelerometer = accelerometer
        self._gps = gps
        self._detector = detector
        self._fusion = fusion_service
        self._event_bus = event_bus
        self._config = config
    
    async def process_frame(self) -> Pothole | None:
        """Process a single frame through the detection pipeline"""
        
        # 1. Capture sensor data
        frame = self._camera.capture_frame()
        accel_data = self._accelerometer.read_acceleration()
        lat, lon = self._gps.get_coordinates()
        
        # 2. Run ML detection
        detections = self._detector.detect(frame)
        
        # 3. Apply multimodal fusion
        result = self._fusion.fuse(
            visual_detections=detections,
            acceleration_data=accel_data,
            min_confidence=self._config.min_confidence,
            accel_threshold=self._config.accel_threshold
        )
        
        if result.is_pothole_detected:
            # 4. Create domain entity
            pothole = Pothole(
                latitude=lat,
                longitude=lon,
                severity=Severity.from_metrics(
                    accel_peak=result.accel_peak,
                    confidence=result.confidence,
                    bbox_area=result.bbox_area
                ),
                confidence=result.confidence,
                accel_peak=result.accel_peak,
                bbox_area=result.bbox_area
            )
            
            # 5. Publish event (loosely coupled)
            await self._event_bus.publish(
                PotholeDetectedEvent(
                    pothole=pothole,
                    raw_frame=frame
                )
            )
            
            return pothole
        
        return None
```

---

### 4. Presentation Layer (APIs & UIs)

This layer handles all external communication - REST APIs, WebSockets, CLI, and future mobile app integration.

```
presentation/
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── potholes.py          # CRUD for potholes
│   │   ├── alerts.py            # Alert endpoints
│   │   ├── sensors.py           # Sensor status endpoints
│   │   └── health.py            # Health check
│   ├── middleware/
│   │   ├── authentication.py    # Future: JWT auth
│   │   └── rate_limiting.py
│   ├── schemas/
│   │   ├── pothole_schema.py    # Pydantic schemas
│   │   └── alert_schema.py
│   └── app.py                   # FastAPI application
├── websocket/
│   ├── server.py                # Real-time updates
│   └── handlers/
│       ├── alert_handler.py
│       └── detection_handler.py
├── cli/
│   ├── main.py                  # CLI entry point
│   └── commands/
│       ├── start.py
│       ├── calibrate.py
│       └── report.py
└── web/                         # Future: Dashboard
    ├── templates/
    └── static/
```

---

## Directory Structure (Complete)

```
pothole_detection_system/
├── README.md
├── pyproject.toml               # Project configuration
├── requirements.txt
├── .env.example
├── docker-compose.yml           # For development
│
├── src/
│   ├── __init__.py
│   │
│   ├── infrastructure/          # Layer 1: External integrations
│   │   ├── __init__.py
│   │   ├── sensors/
│   │   ├── ml/
│   │   ├── persistence/
│   │   ├── alerts/
│   │   └── networking/
│   │
│   ├── domain/                  # Layer 2: Business logic
│   │   ├── __init__.py
│   │   ├── entities/
│   │   ├── value_objects/
│   │   ├── services/
│   │   ├── events/
│   │   └── specifications/
│   │
│   ├── application/             # Layer 3: Use cases
│   │   ├── __init__.py
│   │   ├── services/
│   │   ├── use_cases/
│   │   ├── events/
│   │   ├── dto/
│   │   └── config/
│   │
│   └── presentation/            # Layer 4: APIs & UIs
│       ├── __init__.py
│       ├── api/
│       ├── websocket/
│       ├── cli/
│       └── web/
│
├── tests/
│   ├── unit/
│   │   ├── domain/
│   │   ├── application/
│   │   └── infrastructure/
│   ├── integration/
│   └── e2e/
│
├── data/
│   ├── models/                  # Trained ML models
│   ├── images/                  # Captured images
│   └── database/                # SQLite files
│
├── config/
│   ├── development.yaml
│   ├── production.yaml
│   └── testing.yaml
│
├── scripts/
│   ├── setup_pi.sh              # Raspberry Pi setup
│   ├── train_model.py
│   └── collect_data.py
│
└── docs/
    ├── architecture/
    ├── api/
    └── deployment/
```

---

## Future Civic Platform Extensibility

The architecture is designed to easily extend to a broader civic reporting platform:

### Extension Points for Future Platform

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      CIVIC INFRASTRUCTURE REPORTING PLATFORM                      │
│                           (Future Extension Architecture)                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  CURRENT SYSTEM                          FUTURE EXTENSIONS                        │
│  ─────────────────                       ──────────────────                       │
│                                                                                   │
│  ┌─────────────────┐                     ┌─────────────────┐                     │
│  │  Pothole        │  ─────────────────▶ │ Issue Handler   │                     │
│  │  Detection      │     (Same Base)     │ Factory         │                     │
│  │  Service        │                     │                 │                     │
│  └─────────────────┘                     │ - PotholeHandler│                     │
│                                          │ - StreetlightH  │                     │
│                                          │ - WaterLeakH    │                     │
│                                          │ - GarbageH      │                     │
│                                          │ - WireCutH      │                     │
│                                          └─────────────────┘                     │
│                                                                                   │
│  ┌─────────────────┐                     ┌─────────────────┐                     │
│  │  Local SQLite   │  ─────────────────▶ │ Cloud Database  │                     │
│  │  Repository     │    (Same Interface) │                 │                     │
│  └─────────────────┘                     │ - Firebase      │                     │
│                                          │ - PostgreSQL    │                     │
│                                          │ - MongoDB       │                     │
│                                          └─────────────────┘                     │
│                                                                                   │
│  ┌─────────────────┐                     ┌─────────────────┐                     │
│  │  Buzzer/LED     │  ─────────────────▶ │ Multi-Channel   │                     │
│  │  Alerts         │    (Same Interface) │ Notifications   │                     │
│  └─────────────────┘                     │                 │                     │
│                                          │ - Push (FCM)    │                     │
│                                          │ - SMS           │                     │
│                                          │ - Email         │                     │
│                                          │ - WhatsApp      │                     │
│                                          └─────────────────┘                     │
│                                                                                   │
│  ┌─────────────────┐                     ┌─────────────────┐                     │
│  │  Single Device  │  ─────────────────▶ │ Multi-User      │                     │
│  │  CLI            │     (Extension)     │ Platform        │                     │
│  └─────────────────┘                     │                 │                     │
│                                          │ - Auth System   │                     │
│                                          │ - RBAC          │                     │
│                                          │ - User Profiles │                     │
│                                          │ - Voting        │                     │
│                                          └─────────────────┘                     │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Abstract Issue Type Handler (Open/Closed Principle)

```python
# domain/entities/issue.py - Future base class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class IssueSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class IssueStatus(Enum):
    REPORTED = "REPORTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"

@dataclass
class CivicIssue(ABC):
    """Base class for all civic issues - Open for extension"""
    
    id: str
    latitude: float
    longitude: float
    severity: IssueSeverity
    status: IssueStatus
    reported_at: datetime
    reported_by: str
    image_url: Optional[str] = None
    description: Optional[str] = None
    upvotes: int = 0
    
    @property
    @abstractmethod
    def issue_type(self) -> str:
        """Return the type of civic issue"""
        pass
    
    @abstractmethod
    def determine_severity(self, **kwargs) -> IssueSeverity:
        """Each issue type implements its own severity logic"""
        pass
    
    @abstractmethod
    def get_alert_radius_meters(self) -> int:
        """Alert radius varies by issue type"""
        pass

# Example: Pothole becomes a CivicIssue subclass
@dataclass
class PotholeIssue(CivicIssue):
    accel_peak: float = 0.0
    bbox_area: int = 0
    confidence: float = 0.0
    
    @property
    def issue_type(self) -> str:
        return "POTHOLE"
    
    def determine_severity(self, **kwargs) -> IssueSeverity:
        # Existing severity logic
        if self.accel_peak > 2.5 and self.confidence > 0.7:
            return IssueSeverity.HIGH
        elif self.accel_peak > 1.8:
            return IssueSeverity.MEDIUM
        return IssueSeverity.LOW
    
    def get_alert_radius_meters(self) -> int:
        radii = {
            IssueSeverity.HIGH: 100,
            IssueSeverity.MEDIUM: 50,
            IssueSeverity.LOW: 20
        }
        return radii.get(self.severity, 20)

# Future: New issue types just need to extend CivicIssue
@dataclass
class ElectricWireCutIssue(CivicIssue):
    is_live_wire: bool = False
    weather_condition: str = "clear"
    
    @property
    def issue_type(self) -> str:
        return "ELECTRIC_WIRE_CUT"
    
    def determine_severity(self, **kwargs) -> IssueSeverity:
        if self.is_live_wire:
            return IssueSeverity.CRITICAL
        if self.weather_condition == "rain":
            return IssueSeverity.HIGH
        return IssueSeverity.MEDIUM
    
    def get_alert_radius_meters(self) -> int:
        # Larger radius for dangerous issues
        return 500 if self.severity == IssueSeverity.CRITICAL else 200
```

---

## Benefits of This Architecture

| Benefit | How It's Achieved |
|---------|-------------------|
| **Maintainability** | Clear separation of concerns, single responsibility per module |
| **Testability** | Dependency injection allows mocking any component |
| **Extensibility** | Add new sensors, issue types, or alert channels without modifying core |
| **Scalability** | Event-driven design allows async processing and future distribution |
| **Portability** | Infrastructure layer can be swapped (Pi → Cloud, SQLite → PostgreSQL) |
| **Team Collaboration** | Each layer can be developed independently |

---

## Quick Start Implementation Order

1. **Phase 1: Domain Layer** - Define core entities and business rules
2. **Phase 2: Infrastructure Layer** - Implement sensor adapters and SQLite repository
3. **Phase 3: Application Layer** - Build detection service and event bus
4. **Phase 4: Presentation Layer** - Add REST API and CLI
5. **Phase 5: Testing** - Unit, integration, and end-to-end tests

---

## Next Steps

1. **Approve this architecture** or request modifications
2. **Create the initial project structure**
3. **Implement domain entities first** (no dependencies)
4. **Build sensor adapters** with mock implementations for testing
5. **Implement the detection pipeline**
6. **Add alerting and persistence**
7. **Create REST API and CLI**

> [!IMPORTANT]
> This architecture is designed to grow. You can start simple and add complexity only when needed. The key is maintaining the layer boundaries and using interfaces at integration points.

Do you want me to proceed with implementing this architecture for your project?
