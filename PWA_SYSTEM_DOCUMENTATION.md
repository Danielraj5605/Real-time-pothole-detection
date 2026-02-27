# Real-Time Pothole Alert System — PWA Technical Documentation

> **For IEEE Paper Reference** — This document describes the architecture, data pipeline, alerting mechanism, and design decisions of the Progressive Web App (PWA) driver-alert subsystem.

---

## 1. System Overview

The Pothole Alert System is a real-time civic infrastructure monitoring platform. It consists of three major subsystems:

| Subsystem | Technology | Purpose |
|-----------|-----------|---------|
| **Detection Unit** | Raspberry Pi + YOLOv8 + IMU | Captures video, detects potholes, classifies severity |
| **Backend API** | FastAPI + PostgreSQL (Neon) | Stores detected issues, serves spatial queries |
| **Driver Alert PWA** | React + Leaflet.js | Displays map, tracks GPS, triggers proximity alerts |

```
┌──────────────────┐     HTTP POST       ┌───────────────────┐     HTTP GET        ┌──────────────────┐
│  Detection Unit  │ ──────────────────► │   FastAPI Server  │ ◄────────────────── │   Driver's PWA   │
│  (Raspberry Pi)  │   /api/issues       │   (Neon Postgres) │   /api/issues/nearby│   (Phone/Browser)│
│  YOLOv8 + IMU    │                     │                   │                     │   React + Leaflet│
└──────────────────┘                     └───────────────────┘                     └──────────────────┘
                                                                                          │
                                                                                   GPS + Proximity
                                                                                   Check (Client-Side)
                                                                                          │
                                                                                    ┌─────▼─────┐
                                                                                    │  ALERT!   │
                                                                                    │ Push Notif │
                                                                                    │ Vibration  │
                                                                                    │ Audio Beep │
                                                                                    └───────────┘
```

---

## 2. Data Pipeline (End-to-End)

### Stage 1: Detection & Classification
- The onboard Raspberry Pi captures video at 720p using the Pi Camera Module v2.
- YOLOv8 analyzes frames in real-time to detect potholes.
- Severity is classified based on detection confidence and bounding-box area ratio:

| Area Ratio | Confidence | Severity |
|------------|-----------|----------|
| > 0.10 | > 0.80 | **HIGH** |
| > 0.05 | > 0.60 | **MEDIUM** |
| ≤ 0.05 | Any | **LOW** |

### Stage 2: Data Upload
- Each detection is uploaded to the backend via HTTP `POST /api/issues`:

```json
{
  "type": "pothole",
  "latitude": 12.8546,
  "longitude": 80.068,
  "severity": "HIGH",
  "confidence": 0.92,
  "metadata_info": {
    "area_ratio": 0.12,
    "source": "yolo_detection"
  }
}
```

### Stage 3: Storage
- Stored in a cloud-hosted **PostgreSQL** database (Neon) with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-incremented unique ID |
| `type` | String(50) | Issue type (e.g., `pothole`) |
| `latitude` | Float | GPS latitude |
| `longitude` | Float | GPS longitude |
| `severity` | String(20) | `HIGH`, `MEDIUM`, or `LOW` |
| `confidence` | Float | YOLOv8 detection confidence (0–1) |
| `status` | String(20) | `REPORTED`, `VERIFIED`, or `FIXED` |
| `metadata_info` | JSON | Additional data (area ratio, source) |
| `created_at` | Timestamp | Auto-set on creation |

### Stage 4: Driver Query & Alert
- The PWA polls `GET /api/issues/nearby?lat=X&lon=Y&radius=2000` every 5 seconds.
- The server returns all issues within a 2 km radius.
- The client performs local proximity checking (see Section 4).

---

## 3. PWA Architecture

### 3.1 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI Framework | React 19 | Component-based SPA |
| Routing | React Router v7 | Multi-page navigation |
| Map Rendering | Leaflet.js + react-leaflet | Interactive map with OpenStreetMap tiles |
| Styling | Pure CSS (no frameworks) | Custom responsive design |
| Notifications | Browser Notification API | System-level push notifications |
| Vibration | Vibration API | Haptic feedback on mobile |
| Audio | Web Audio API | Severity-based alert beeps |
| GPS | Geolocation API | Real-time device location tracking |
| Build Tool | Vite | Fast dev server and production bundler |

### 3.2 File Structure

```
webapp-react/
├── index.html                    # Entry point, PWA metadata
├── vite.config.js                # Build configuration
├── package.json                  # Dependencies
└── src/
    ├── main.jsx                  # React DOM root
    ├── index.css                 # Global styles (light theme)
    ├── App.jsx                   # Router shell
    ├── components/
    │   └── Navbar.jsx            # Navigation bar
    ├── pages/
    │   ├── Dashboard.jsx         # Main map + sidebar + alerts
    │   └── About.jsx             # Project information
    └── services/
        └── notifications.js      # Push notification logic
```

### 3.3 Dashboard Features

1. **Real-Time GPS Tracking** — Uses `navigator.geolocation.watchPosition()` with high-accuracy mode.
2. **Map Visualization** — OpenStreetMap tiles with color-coded pothole markers:
   - 🔴 Red = HIGH severity
   - 🟡 Orange/Yellow = MEDIUM severity
   - 🟢 Green = LOW severity
3. **50m Alert Radius** — A dashed blue circle around the user shows the alert zone.
4. **Sidebar Stats** — Live count of HIGH/MEDIUM/LOW issues in the vicinity.
5. **Pothole List** — Scrollable list of nearby issues with coordinates and confidence scores.

---

## 4. Distributed Alerting Architecture (No Server Overload)

### 4.1 The Problem

If N drivers continuously upload GPS coordinates and the server must check each driver against M potholes, the computational complexity is **O(N × M)** per second — unsustainable at scale.

### 4.2 Our Solution: Client-Side Proximity Checking

The computation is **offloaded to each driver's device**. The server is only responsible for spatial queries, not alert logic.

```
                  ❌ Naive Approach (O(N × M))
    ┌────────────────────────────────────────────┐
    │  Server receives GPS from 1000 drivers     │
    │  Server checks each against 5000 potholes  │
    │  = 5,000,000 Haversine calculations/sec    │
    │  Server sends push alerts                  │
    │  Result: SERVER OVERLOAD                   │
    └────────────────────────────────────────────┘

                  ✅ Our Approach (O(M) per client)
    ┌────────────────────────────────────────────┐
    │  Driver's phone sends: "What's near me?"   │
    │  Server returns 5–10 nearby potholes       │
    │  Phone checks: "Am I within 50m?"          │
    │  Phone triggers alert locally              │
    │  Result: ZERO server load for alerting     │
    └────────────────────────────────────────────┘
```

### 4.3 Complexity Analysis

| Metric | Server-Side (Naive) | Client-Side (Ours) |
|--------|--------------------|--------------------|
| Server computation per driver | O(M) every second | O(M) every 5 seconds |
| Alert computation | Server does all | Each phone does its own |
| Network traffic | Continuous GPS uploads | One GET request per 5s |
| Scalability | Degrades with N drivers | Linear, independent |
| Real-time accuracy | Limited by upload rate | Phone GPS at 10Hz |
| Push notification latency | Server → FCM → phone | Instant (local) |

### 4.4 Haversine Distance Calculation

Both client and server use the Haversine formula for distance:

```
a = sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)
distance = 2R × atan2(√a, √(1−a))

Where R = 6,371,000 meters (Earth's radius)
```

This is computed on the client for each of the ~5–10 nearby potholes returned by the server, resulting in negligible computational cost.

---

## 5. Notification System

### 5.1 Multi-Modal Alerts

When a driver enters within **50 meters** of a pothole, three simultaneous alerts are triggered:

| Mode | API Used | Behavior |
|------|----------|----------|
| **Push Notification** | Notification API | System-level banner (works in background) |
| **Vibration** | Vibration API | Severity-dependent pattern |
| **Audio Beep** | Web Audio API | Tone frequency varies by severity |

### 5.2 Vibration Patterns

```
HIGH:   ━━━━ ▪ ━━━━ ▪ ━━━━ ▪ ━━━━   (300ms on, 100ms off, ×4)
MEDIUM: ━━━ ▪ ━━━ ▪ ━━━              (200ms on, 100ms off, ×3)
LOW:    ━━ ▪ ━━                       (150ms on, 100ms off, ×2)
```

### 5.3 Alert Deduplication

To prevent notification spam, the system tracks which pothole IDs have already been alerted:

1. When a pothole enters the 50m radius → trigger alert, add ID to `alertedSet`
2. Same pothole within 50m again → **no duplicate alert**
3. Driver moves >200m away from that pothole → remove ID from `alertedSet`
4. Driver returns within 50m → fresh alert triggers again

---

## 6. Backend API

### 6.1 Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Server health check |
| `POST` | `/api/issues` | Report a new pothole detection |
| `GET` | `/api/issues/nearby?lat=X&lon=Y&radius=R` | Query potholes within R meters |

### 6.2 CORS

The backend uses FastAPI's `CORSMiddleware` with `allow_origins=["*"]` for development, enabling the PWA to communicate with the API from any origin.

### 6.3 Database Connection

- **Provider**: Neon (serverless PostgreSQL)
- **ORM**: SQLAlchemy 2.0
- **Connection**: Pooled via Neon's connection pooler endpoint
- **Benefit**: Zero infrastructure management, auto-scaling, always available

---

## 7. Future Enhancements

| Enhancement | Technology | Impact |
|-------------|-----------|--------|
| Spatial indexing | PostGIS + `ST_DWithin` | O(log M) server queries instead of O(M) |
| Real-time updates | WebSockets | Instant new-pothole notifications |
| Offline support | Service Worker + Cache API | Works without internet |
| Background alerts | Firebase Cloud Messaging | Alerts when app is closed |
| Route planning | OSRM / Google Directions API | "Safest route" avoiding potholes |
| Crowdsource reporting | Photo upload + GPS | Users report potholes directly |

---

## 8. Key Design Decisions (for Paper Discussion)

1. **Client-side alerting over server-side**: Eliminates the N×M bottleneck. Each phone independently queries and checks proximity, making the system linearly scalable.

2. **Polling over WebSockets**: For a prototype with <100 concurrent users, HTTP polling every 5 seconds is simpler and sufficient. WebSockets would be the upgrade path for production.

3. **PWA over native app**: No app store deployment required. Users access via URL. Supports GPS, notifications, and vibration through standard Web APIs. Works on both Android and iOS.

4. **Cloud PostgreSQL over local DB**: Neon's serverless PostgreSQL provides zero-config hosting with automatic scaling, ideal for a prototype that needs reliability without DevOps overhead.

5. **Haversine over Vincenty**: Haversine is accurate to ~0.3% for short distances (<10 km), computationally cheaper, and sufficient for 50m proximity checks. Vincenty's iterative solution is unnecessary for this use case.

---

## 9. Performance Characteristics

| Metric | Value |
|--------|-------|
| GPS update frequency | ~1 Hz (phone default) |
| Backend polling interval | 5 seconds |
| Alert detection latency | < 5 seconds |
| Notification delivery | Instant (client-side) |
| Haversine computation time | < 0.01ms per pair |
| Memory footprint (browser) | ~15 MB |
| Network per poll cycle | ~2 KB (JSON response) |

---

*Document prepared for IEEE paper reference — Pothole Alert System, February 2026*
