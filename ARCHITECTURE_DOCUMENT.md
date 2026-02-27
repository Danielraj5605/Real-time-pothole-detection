# Architecture Document

## Real-Time Pothole Detection and Alert System

**Version:** 1.0.0 | **Date:** February 2026

---

## 1. Application Architecture

### 1.1 Architecture Comparison

| Criteria | Microservices | Event-Driven | Serverless |
|---|---|---|---|
| Suitability | ✅ High — system naturally splits into Edge, Backend, and Frontend services | ⚠️ Partial — detection events fit, but a message broker adds unnecessary overhead | ❌ Low — detection engine needs persistent, long-running GPU/CPU processing |
| Independent Deployment | Each service uses a different tech stack and deploys independently | Requires a central broker (Kafka/RabbitMQ) coupling all services | Cold starts and time limits conflict with real-time video processing |
| Scalability | Backend API can scale horizontally as reports grow | Good for high-throughput event streams, overkill for current scale | Auto-scales but not suited for continuous sensor processing |
| Complexity | Manageable without Kubernetes for a small team | Adds broker management overhead | Stateless functions can't maintain object tracking state across frames |

### 1.2 Chosen Architecture: Lightweight Microservices

The system adopts a **lightweight microservices architecture** with three independently deployable services:

```
┌─────────────────────┐   REST API (POST)   ┌──────────────────┐
│  EDGE SERVICE       │────────────────────→│  CLOUD BACKEND   │
│  (Raspberry Pi)     │                     │  (FastAPI)       │
│                     │                     │                  │
│  • Pi Camera        │                     │  • Issue CRUD    │
│  • MPU-9250 IMU     │                     │  • Proximity     │
│  • YOLOv8 Pipeline  │                     │    Search        │
└─────────────────────┘                     └────────┬─────────┘
                                                     │ SQL
                                            ┌────────▼─────────┐
                                            │  PostgreSQL      │
                                            │  (Neon Cloud)    │
                                            └────────┬─────────┘
                                                     │
┌─────────────────────┐   REST API (GET)    ┌────────▼─────────┐
│  CLIENT PWA         │◄───────────────────│  CLOUD BACKEND   │
│  (React + Vite)     │   every 5 sec      │  (FastAPI)       │
│                     │                     └──────────────────┘
│  • Leaflet Map      │
│  • Smartphone GPS   │
│  • Proximity Alerts │
└─────────────────────┘
```

**Why this architecture:**

1. **Natural service boundaries** — Edge (Raspberry Pi), Cloud (FastAPI), and Client (React PWA) are inherently separate with distinct tech stacks.
2. **Independent updates** — The ML model can be retrained without touching the backend; the PWA can be redeployed without server changes.
3. **Simple scaling** — The backend API can scale horizontally as the number of reports grows.
4. **Event-driven patterns built-in** — The PWA polls the API every 5 seconds and triggers alerts client-side, without needing a message broker.

---

## 2. Database

### 2.1 ER Diagram

The system uses a single-entity design centered around the **Issue** table:

```
┌────────────────────────────────────────┐
│               ISSUE                     │
├────────────────────────────────────────┤
│  PK  id            INTEGER (auto)      │
│      type          VARCHAR(50) NOT NULL │
│      latitude      FLOAT NOT NULL      │
│      longitude     FLOAT NOT NULL      │
│      severity      VARCHAR(20)         │
│      confidence    FLOAT               │
│      status        VARCHAR(20)         │
│      image_url     TEXT                │
│      description   TEXT                │
│      metadata_info JSON                │
│      created_at    TIMESTAMPTZ         │
│      updated_at    TIMESTAMPTZ         │
└────────────────────────────────────────┘
```

### 2.2 Schema Design

**Database:** PostgreSQL (hosted on Neon Cloud — serverless PostgreSQL)  
**ORM:** SQLAlchemy (Python)

```sql
CREATE TABLE issues (
    id            SERIAL PRIMARY KEY,
    type          VARCHAR(50)   NOT NULL,       -- 'pothole'
    latitude      FLOAT         NOT NULL,       -- GPS latitude
    longitude     FLOAT         NOT NULL,       -- GPS longitude
    severity      VARCHAR(20),                  -- 'LOW', 'MEDIUM', 'HIGH'
    confidence    FLOAT,                        -- ML confidence (0.0 - 1.0)
    status        VARCHAR(20)   DEFAULT 'REPORTED', -- REPORTED → VERIFIED → FIXED
    image_url     TEXT,
    description   TEXT,
    metadata_info JSONB,                        -- Extensible JSON metadata
    created_at    TIMESTAMPTZ   DEFAULT NOW(),
    updated_at    TIMESTAMPTZ
);
```

**Design Rationale:**
- **Single table** keeps queries simple; no JOINs needed for the current use case.
- **JSONB `metadata_info`** allows flexible storage of detection metadata (source, area ratio, depth) without schema changes.
- **Status lifecycle:** `REPORTED` → `VERIFIED` → `FIXED` tracks the issue resolution workflow.

---

## 3. Data Exchange Contract

### 3.1 Frequency of Data Exchanges

| Exchange | Direction | Frequency |
|---|---|---|
| Pothole Upload | Edge Device → Backend | On each confirmed detection (during drive) |
| Nearby Issues Query | PWA → Backend | Every **5 seconds** (polling) |
| Camera Frames | Camera → Detection Engine | **30 FPS** continuous |
| Accelerometer Data | MPU-9250 → Detection Engine | **50 Hz** continuous |
| Smartphone GPS | Browser Geolocation API → PWA | ~**1 Hz** (device default) |

### 3.2 Data Sets

**Detection Upload (Edge → Backend):**

```json
{
    "type": "pothole",
    "latitude": 12.8546,
    "longitude": 80.0680,
    "severity": "HIGH",
    "confidence": 0.87,
    "description": "Pothole detected on main road.",
    "metadata_info": {
        "source": "detection_engine",
        "area_ratio": 0.085
    }
}
```

**Nearby Issues Response (Backend → PWA):**

```json
[
    {
        "id": 42,
        "type": "pothole",
        "latitude": 12.8548,
        "longitude": 80.0682,
        "severity": "HIGH",
        "confidence": 0.87,
        "status": "REPORTED",
        "created_at": "2026-02-27T10:15:30+05:30"
    }
]
```

### 3.3 Mode of Exchanges

| Exchange Path | Mode | Protocol | Format |
|---|---|---|---|
| Edge Device → Backend API | **REST API** | HTTP POST | JSON |
| PWA → Backend API | **REST API** | HTTP GET | JSON |
| Camera → Detection Engine | **Hardware (CSI/USB)** | Direct | Video frames |
| MPU-9250 → Detection Engine | **I2C Bus** | I2C (0x68) | Raw sensor values |
| Smartphone → PWA | **Browser API** | Geolocation API | Lat/Lon coordinates |
| Backend → Database | **ORM Connection** | TCP/SSL | SQL via SQLAlchemy |

**API Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/issues` | Report a new pothole issue |
| `GET` | `/api/issues/nearby?lat=&lon=&radius=` | Find issues within radius (meters) |

---

## 4. System Diagrams

### 4.1 Use Case Diagram

```
                    ┌──────────────────────────────────────────┐
                    │      Pothole Detection System            │
                    │                                          │
                    │   ┌──────────────────────────────┐       │
  Edge Device ──────┤──→│ Detect Potholes (YOLOv8)     │       │
  (Raspberry Pi)    │   └──────────┬───────────────────┘       │
                    │              │ «includes»                │
                    │   ┌──────────▼───────────────────┐       │
                    │   │ Classify Severity             │       │
                    │   └──────────┬───────────────────┘       │
                    │              │ «includes»                │
                    │   ┌──────────▼───────────────────┐       │
                    │   │ Upload Detection to Server    │       │
                    │   └──────────────────────────────┘       │
                    │                                          │
                    │   ┌──────────────────────────────┐       │
  Driver ───────────┤──→│ View Pothole Map (Dashboard)  │       │
  (Smartphone)      │   └──────────────────────────────┘       │
                    │   ┌──────────────────────────────┐       │
                    ├──→│ Receive Proximity Alerts      │       │
                    │   └──────────────────────────────┘       │
                    │   ┌──────────────────────────────┐       │
                    ├──→│ Report Pothole Manually       │       │
                    │   └──────────────────────────────┘       │
                    │                                          │
                    │   ┌──────────────────────────────┐       │
  Developer ────────┤──→│ Train Detection Model         │       │
                    │   └──────────────────────────────┘       │
                    │                                          │
                    │   ┌──────────────────────────────┐       │
  Authority ────────┤──→│ View Statistics & Reports     │       │
                    │   └──────────────────────────────┘       │
                    └──────────────────────────────────────────┘
```

### 4.2 Class Diagram

```
┌──────────────────────────┐      ┌─────────────────────────┐
│       Config             │      │     YOLODetector         │
├──────────────────────────┤      ├─────────────────────────┤
│ camera_id: int           │      │ model_path: str         │
│ camera_width: int        │      │ confidence: float       │
│ model_path: str          │      │ model: YOLO             │
│ confidence_threshold: flt│      ├─────────────────────────┤
│ enable_tracking: bool    │      │ initialize()            │
│ enable_classification: bl│      │ detect(frame): List     │
└──────────┬───────────────┘      └───────────┬─────────────┘
           │ uses                              │ uses
           ▼                                   ▼
┌──────────────────────────────────────────────────────────┐
│                   DetectionPipeline                       │
├──────────────────────────────────────────────────────────┤
│ detector: YOLODetector                                    │
│ config: Config                                            │
│ tracked_objects: Dict                                     │
├──────────────────────────────────────────────────────────┤
│ 1. clean_frame(frame)     → Preprocessed frame           │
│ 2. find_objects(frame)    → Bounding boxes                │
│ 3. track_objects(dets)    → Tracked objects with IDs      │
│ 4. isolate(frame, bbox)  → Cropped pothole region        │
│ 5. read_info(frame, bbox) → Features (area, texture)     │
│ 6. identify(features)    → Severity (LOW/MED/HIGH)       │
│    process(frame)         → List[PotholeInfo]             │
└──────────────────────────────────────────────────────────┘
           │ produces
           ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│     PotholeInfo          │     │     TrackedObject        │
├──────────────────────────┤     ├──────────────────────────┤
│ track_id: int            │     │ track_id: int            │
│ bbox: Tuple              │     │ bbox_history: deque      │
│ confidence: float        │     │ first_seen: int          │
│ severity: str            │     │ last_seen: int           │
│ area: int                │     │ frames_missing: int      │
│ depth: str               │     │ is_active: bool          │
└──────────────────────────┘     └──────────────────────────┘

┌──────────────────────────┐
│  Issue (DB Model)        │
├──────────────────────────┤
│ id: int (PK)             │
│ type: str                │     ┌──────────────────────────┐
│ latitude: float          │     │  IssueCreate (API Input) │
│ longitude: float         │     ├──────────────────────────┤
│ severity: str            │     │ type: str                │
│ confidence: float        │     │ latitude: float          │
│ status: str              │     │ longitude: float         │
│ metadata_info: JSON      │     │ severity: str            │
│ created_at: datetime     │     │ confidence: float        │
│ updated_at: datetime     │     │ description: str         │
└──────────────────────────┘     └──────────────────────────┘
```

### 4.3 Data Flow Diagram (DFD)

**Level 0 — Context Diagram:**

```
  Camera + MPU-9250             Pothole Detection           PostgreSQL
  (Sensor Data)       ───→      System             ───→     (Cloud DB)
                                    │
                                    │  Alerts + Map
                                    ▼
                               Driver / User
                              (Smartphone PWA)
```

**Level 1 — Detailed DFD:**

```
┌─── Edge Device (Raspberry Pi) ───────────────────────────────────────┐
│                                                                       │
│  Camera ──→ [1. Clean Frame] ──→ [2. Find Objects (YOLOv8)]         │
│                                          │                            │
│                                   [3. Track Objects (IoU)]           │
│                                          │                            │
│                                   [4. Isolate Region]                │
│                                          │                            │
│  MPU-9250 ──────────────────→ [5. Read Information]                  │
│                                          │                            │
│                                   [6. Identify Severity]             │
│                                          │                            │
└──────────────────────────────────────────┼───────────────────────────┘
                                           │ POST /api/issues
                                           ▼
                                    ┌─────────────┐
                                    │ FastAPI      │──→ PostgreSQL (Neon)
                                    │ Backend      │
                                    └──────┬──────┘
                                           │ GET /api/issues/nearby
                                           ▼
                                    ┌─────────────┐
                                    │ React PWA   │──→ Smartphone GPS
                                    │ (Dashboard) │──→ Alerts (Sound/Vibration)
                                    └─────────────┘
```

### 4.4 Component Diagram

```
┌─── Edge Layer ────────────────┐  ┌─── Cloud Layer ────────────┐
│                                │  │                             │
│  [Pi Camera] ──┐               │  │  [FastAPI Server]           │
│                ▼               │  │       │                     │
│  [MPU-9250] ──→ [Detection    │  │  [SQLAlchemy ORM]           │
│                  Engine]      │  │       │                     │
│                  │             │  │  [Neon PostgreSQL]          │
│                  │ POST        │  │                             │
│                  ├────────────────→                             │
│  [Local SQLite]◄─┘             │  └─────────────┬──────────────┘
│                                │                │ GET
└────────────────────────────────┘                ▼
                                   ┌─── Client Layer ────────────┐
                                   │                              │
                                   │  [React App]                 │
                                   │     ├── [Dashboard + Map]    │
                                   │     ├── [Notification Svc]   │
                                   │     └── [Geolocation API]    │
                                   │         (Smartphone GPS)     │
                                   └──────────────────────────────┘
```

### 4.5 Sequence Diagram

**Pothole Detection and Upload:**

```
Camera      Detection Engine    YOLOv8     Pipeline    FastAPI    PostgreSQL
  │              │                │           │           │           │
  │─ frame ─────→│                │           │           │           │
  │              │─ inference ───→│           │           │           │
  │              │◄─ detections ──│           │           │           │
  │              │─── track + classify ─────→│           │           │
  │              │◄── PotholeInfo (severity) ─│           │           │
  │              │───── POST /api/issues ────────────────→│           │
  │              │                │           │           │─ INSERT ─→│
  │              │                │           │           │◄── OK ────│
  │              │◄────────────── 200 OK ────────────────│           │
```

**PWA Proximity Alert:**

```
User (Phone)    React PWA       Geolocation API    FastAPI    PostgreSQL
     │              │                  │               │           │
     │─ open app ──→│                  │               │           │
     │              │─ watchPosition ─→│               │           │
     │              │◄─ lat, lon ──────│               │           │
     │              │                  │               │           │
     │              │── GET /nearby?lat=X&lon=Y ─────→│           │
     │              │                  │               │─ SELECT ─→│
     │              │                  │               │◄── rows ──│
     │              │◄── nearby issues (JSON) ────────│           │
     │              │                  │               │           │
     │              │─ check distance ─│               │           │
     │     ◄── 🔔 Notification + 📳 Vibration + 🔊 Sound        │
     │              │                  │               │           │
     │      (repeats every 5 seconds)  │               │           │
```

### 4.6 Deployment Diagram

```
┌─── Vehicle (Edge Node) ──────────────────────────────────────────┐
│                                                                    │
│   Raspberry Pi Zero WH (ARM11, 512MB RAM)                         │
│   ├── Pi Camera Module (CSI, 640×480 @ 30 FPS)                    │
│   ├── MPU-9250 IMU (I2C, 50 Hz)                                  │
│   ├── Python 3.9+ (PyTorch, Ultralytics, OpenCV)                  │
│   └── Local SQLite Database                                       │
│                                                                    │
└──────────────────────┬────────────────────────────────────────────┘
                       │ HTTPS (REST API)
                       ▼
┌─── Cloud Server ─────────────────┐    ┌─── Neon Cloud ───────────┐
│                                   │    │                          │
│  FastAPI + Uvicorn                │───→│  PostgreSQL (Serverless) │
│  (Python 3.9+)                    │    │  Connection Pooling      │
│                                   │    │  SSL/TLS Encrypted       │
└──────────────────────┬────────────┘    └──────────────────────────┘
                       │ HTTPS (REST API)
                       ▼
┌─── End User Devices ─────────────────────────────────────────────┐
│                                                                    │
│   📱 Smartphone Browser (PWA)                                     │
│   ├── GPS via Geolocation API (navigator.geolocation)             │
│   ├── Push Notifications (Notification API)                       │
│   ├── Vibration Alerts (Vibration API)                            │
│   └── Audio Alerts (Web Audio API)                                │
│                                                                    │
│   💻 Desktop Browser (Dashboard view)                             │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Conclusion

The system uses a **lightweight microservices architecture** with three layers: an edge detection engine (Raspberry Pi with YOLOv8 + MPU-9250), a cloud backend (FastAPI + PostgreSQL), and a client PWA (React + Leaflet with smartphone GPS). REST APIs connect all layers. This architecture keeps development simple while allowing each service to be updated and scaled independently.
