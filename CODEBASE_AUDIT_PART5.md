# 🔍 Complete Architectural Audit: Real-time Pothole Detection System
## Part 5: Key Files, Navigation Guide & Takeaways

---

## 🔑 Critical Files (Do Not Break!)

### Tier 1: CRITICAL - System Will Break

| File | Lines | Why Critical |
|------|-------|--------------|
| `src/application/config/dependency_injection.py` | 304 | **The glue that holds everything together**. Wires all components. Breaking this breaks the entire system. |
| `config/config.json` | 303 | **Master configuration**. All system parameters. Invalid config = broken system. |
| `src/domain/entities/pothole.py` | 109 | **Core domain model**. Database schema depends on this. Changes affect persistence. |
| `src/infrastructure/ml/adapters/yolov8_detector.py` | 143 | **ML inference wrapper**. Breaking this breaks all detection. |

### Tier 2: Important - Features Will Break

| File | Lines | Why Important |
|------|-------|---------------|
| `src/application/services/detection_service.py` | 184 | Main pipeline orchestrator. Breaking this breaks detection flow. |
| `src/application/services/live_detection_service.py` | 564 | Real-time camera service. Breaking this breaks live mode. |
| `src/domain/services/fusion_service.py` | 106 | Fusion logic. Breaking this breaks multimodal detection. |
| `src/infrastructure/persistence/adapters/sqlite_repository.py` | 274 | Database operations. Breaking this breaks persistence. |

### Tier 3: Useful - Functionality Will Degrade

| File | Purpose |
|------|---------|
| `src/application/events/event_bus.py` | Event system. Breaking this breaks loose coupling. |
| `src/application/services/alert_service.py` | Alert management. Breaking this breaks alerts. |
| `src/infrastructure/sensors/adapters/opencv_camera.py` | Camera adapter. Breaking this breaks camera support. |

---

## 📊 Core Business Logic Locations

### Where Business Rules Live

| Business Rule | Location | Method/Function |
|---------------|----------|-----------------|
| **Severity Classification** | `domain/entities/pothole.py` | `Severity.from_metrics(accel_peak, confidence, bbox_area)` |
| **Fusion Logic** | `domain/services/fusion_service.py` | `FusionService.fuse(detections, accel_data)` |
| **Proximity Alerts** | `domain/services/proximity_calculator.py` | `ProximityCalculator.haversine_distance()` |
| **Alert Thresholds** | `domain/entities/pothole.py` | `Pothole.should_alert_at_distance(distance_m)` |

### Severity Classification Rules

```python
# From domain/entities/pothole.py
if accel_peak > 2.5 and confidence > 0.7 and bbox_area > 10000:
    return Severity.HIGH
elif accel_peak > 1.8 and confidence > 0.6 and bbox_area > 5000:
    return Severity.MEDIUM
else:
    return Severity.LOW
```

### Fusion Algorithm

```python
# From domain/services/fusion_service.py
vision_score = min(vision_confidence, 1.0)
accel_score = min(accel_peak / 3.0, 1.0)  # Normalize to 0-1

fusion_score = (
    vision_weight * vision_score +
    accel_weight * accel_score
)

is_detected = (
    fusion_score >= fusion_threshold and
    vision_confidence >= min_confidence and
    accel_peak >= accel_threshold
)
```

---

## ⚙️ Configuration Control Points

### Key Config Sections That Control Behavior

| Config Path | Default | What It Controls |
|-------------|---------|------------------|
| `hardware.mode` | "real" | "mock" for testing, "real" for production |
| `detection.min_confidence` | 0.5 | Minimum ML confidence threshold |
| `detection.accel_threshold` | 1.5 | Minimum acceleration threshold (g-force) |
| `fusion.vision_weight` | 0.6 | Weight given to visual detection |
| `fusion.accel_weight` | 0.4 | Weight given to accelerometer |
| `fusion.combined_detection_threshold` | 0.5 | Fusion score threshold |
| `alerts.max_distance_m` | 200 | Alert trigger distance (meters) |
| `live_detection.confidence_threshold` | 0.25 | Live camera detection threshold |
| `live_detection.target_fps` | 15 | Target FPS for live detection |
| `persistence.database_path` | "data/database/potholes.db" | SQLite database location |

### Environment Selection

```bash
# Windows
set POTHOLE_ENV=production
python main.py

# Linux/Mac
export POTHOLE_ENV=production
python main.py
```

**Environments:**
- `development` (default) - Mock hardware, verbose logging
- `production` - Real hardware, optimized settings
- `testing` - Reduced thresholds for testing

---

## 🎯 Where to Look to Modify Features

### Common Modification Scenarios

| I want to... | Look at... | What to change |
|--------------|------------|----------------|
| Change detection thresholds | `config/config.json` | `detection.min_confidence`, `detection.accel_threshold` |
| Modify severity rules | `domain/entities/pothole.py` | `Severity.from_metrics()` method |
| Change fusion weights | `config/config.json` | `fusion.vision_weight`, `fusion.accel_weight` |
| Add new sensor type | `infrastructure/sensors/` | Create new adapter implementing interface |
| Add new alert channel | `infrastructure/alerts/adapters/` | Create new adapter implementing `AlertChannelInterface` |
| Modify ML model | `infrastructure/ml/adapters/yolov8_detector.py` | Change model loading or inference |
| Add new event type | `application/events/` | Create new event class, subscribe handlers |
| Change database schema | `infrastructure/persistence/adapters/sqlite_repository.py` | Modify `_create_tables()` method |
| Adjust camera settings | `config/config.json` | `hardware.camera.*` settings |
| Change visualization | `application/services/live_detection_service.py` | `visualize_detections()` method |

---

## ✅ Safe to Modify

### Configuration Values
- ✅ All values in `config/*.json`
- ✅ Threshold parameters
- ✅ UI/visualization settings (colors, fonts, sizes)
- ✅ Logging levels and formats
- ✅ Database path
- ✅ Model paths

### Adding New Components
- ✅ New sensor adapters (implement interface)
- ✅ New alert channels (implement interface)
- ✅ New event types (extend base event)
- ✅ New event handlers/callbacks
- ✅ New application services

---

## ⚠️ Change with Caution

### Entity Structures
- ⚠️ Changing `Pothole` fields affects database schema
- ⚠️ Changing `Alert` fields affects database schema
- ⚠️ Must migrate existing data if schema changes

### Interface Contracts
- ⚠️ Changing interface signatures breaks all adapters
- ⚠️ Must update all implementations simultaneously

### Event Structures
- ⚠️ Changing event fields breaks subscribers
- ⚠️ Must update all event handlers

---

## ❌ Do Not Change

### Core Architecture
- ❌ `DependencyContainer` initialization order
- ❌ Layer dependencies (must flow inward)
- ❌ Protocol/interface signatures without updating adapters

### Business Logic
- ❌ Core fusion logic without thorough testing
- ❌ Severity classification without validation
- ❌ Distance calculations (Haversine formula)

---

## 🧭 Navigation Guide

### If You Want To...

#### Understand the System
1. Start with `README.md` - High-level overview
2. Read `ARCHITECTURE.md` - Architecture details
3. Review `USAGE_GUIDE.md` - Usage examples
4. Check this audit - Complete codebase understanding

#### Run the Application
1. **Mock Demo:** `python main.py`
2. **Live Camera:** `python live_detection.py`
3. **Architecture Demo:** `python demo_architecture.py`

#### Configure Settings
1. Edit `config/config.json` for base settings
2. Edit `config/production.json` for production overrides
3. Set `POTHOLE_ENV` environment variable

#### Understand Detection Flow
1. Read `application/services/detection_service.py`
2. Check `domain/services/fusion_service.py`
3. Review `infrastructure/ml/adapters/yolov8_detector.py`

#### Add Hardware Support
1. Create interface in `infrastructure/sensors/interfaces/`
2. Create adapter in `infrastructure/sensors/adapters/`
3. Register in `DependencyContainer.get_xxx()` method
4. Configure in `config/config.json`

#### Train a Model
1. Prepare data: `python scripts/prepare_dataset.py`
2. Train: `python scripts/train.py --epochs 100 --batch 16`
3. Evaluate: `python scripts/evaluate.py`
4. Update `config.json` with new model path

#### Debug Issues
1. Check `logs/pothole_detection.log`
2. Check `logs/live_detection.log`
3. Enable DEBUG logging in config
4. Use mock sensors to isolate issues

---

## 🗂️ File Type Reference

### By File Extension

| Extension | Count | Purpose |
|-----------|-------|---------|
| `.py` | 65+ | Python source code |
| `.json` | 4 | Configuration files |
| `.md` | 5+ | Documentation |
| `.yaml` | 1 | YOLO dataset config |
| `.csv` | 10 | Accelerometer data |
| `.jpg` | 2009+ | Training images |
| `.txt` | 2009+ | YOLO labels |
| `.pt` | 1+ | PyTorch model weights |
| `.db` | 1 | SQLite database |
| `.log` | 2+ | Log files |

### By Purpose

| Purpose | Files |
|---------|-------|
| **Entry Points** | `main.py`, `live_detection.py`, `demo_architecture.py` |
| **Configuration** | `config/*.json` |
| **Domain Logic** | `src/domain/**/*.py` |
| **Application Services** | `src/application/services/*.py` |
| **Infrastructure** | `src/infrastructure/**/*.py` |
| **Legacy Code** | `src/vision/*.py`, `src/accelerometer/*.py`, `src/fusion/*.py` |
| **Scripts** | `scripts/*.py` |
| **Documentation** | `*.md` |
| **Data** | `Datasets/**/*`, `data/**/*` |

---

## 🚫 Safe to Ignore (Dead/Deprecated Code)

### Deprecated Modules
1. **`src/fusion/`** - Entire directory
   - ❌ Use `src/domain/services/fusion_service.py` instead
   - Reason: Monolithic, tightly coupled

2. **`demo_legacy.py`** (if exists)
   - ❌ Use `main.py` or `demo_architecture.py` instead

3. **`config.yaml`** (if exists)
   - ❌ Use `config/config.json` instead
   - Reason: Migrated to JSON

### Temporary Files
- `__pycache__/` - Python bytecode cache
- `*.pyc` - Compiled Python files
- `.DS_Store` - macOS metadata
- `Thumbs.db` - Windows thumbnails

---

## 📈 Project Statistics

### Code Metrics
- **Total Python Files:** 65+
- **Total Lines of Code:** ~15,000+
- **Largest File:** `live_detection_service.py` (564 lines)
- **Most Complex:** `dependency_injection.py` (304 lines, 15+ methods)

### Data Metrics
- **Training Images:** 592
- **Validation Images:** 221
- **Total Labels:** 2,009
- **Accelerometer Trips:** 5 datasets
- **Database Tables:** 2 (potholes, alerts)

### Dependencies
- **Core Dependencies:** 11 packages
- **Optional Dependencies:** 4 packages
- **Python Version:** 3.8+

---

## 🎓 Key Takeaways

### Architecture Strengths
✅ **Clean Separation** - Clear layer boundaries  
✅ **Testability** - Mock implementations for all hardware  
✅ **Extensibility** - Easy to add new sensors/alerts  
✅ **Maintainability** - Single responsibility per module  
✅ **Configurability** - JSON-based configuration  
✅ **Event-Driven** - Loose coupling via event bus  

### What Makes This System Production-Ready
1. **Comprehensive error handling** - Try/catch blocks everywhere
2. **Resource management** - Proper cleanup and context managers
3. **Logging** - Detailed logging at all levels
4. **Configuration** - Environment-specific settings
5. **Database** - Persistent storage with indexes
6. **Testing support** - Mock implementations
7. **Documentation** - README, ARCHITECTURE, USAGE_GUIDE

### Migration Path from Legacy
```
Legacy Code → New Architecture

src/fusion/engine.py → domain/services/fusion_service.py
src/vision/detector.py → infrastructure/ml/adapters/yolov8_detector.py
Direct sensor access → infrastructure/sensors/adapters/*
Tight coupling → Event-driven via EventBus
YAML config → JSON config
```

---

## 📞 Quick Reference

### Most Important Files (Top 10)
1. `main.py` - Main entry point
2. `live_detection.py` - Live camera detection
3. `config/config.json` - Master configuration
4. `src/application/config/dependency_injection.py` - DI container
5. `src/application/services/detection_service.py` - Main pipeline
6. `src/domain/entities/pothole.py` - Core entity
7. `src/domain/services/fusion_service.py` - Fusion logic
8. `src/infrastructure/ml/adapters/yolov8_detector.py` - ML wrapper
9. `src/infrastructure/persistence/adapters/sqlite_repository.py` - Database
10. `src/application/events/event_bus.py` - Event system

### Most Frequently Modified
1. `config/config.json` - Tuning parameters
2. `domain/entities/pothole.py` - Business rules
3. `domain/services/fusion_service.py` - Fusion algorithm
4. `application/services/live_detection_service.py` - Visualization

### Never Touch
1. `src/application/config/dependency_injection.py` - Unless you know what you're doing
2. Interface files - Without updating all adapters
3. Database schema - Without migration plan

---

## 🎯 Final Summary

### You Now Know:
✅ What every folder and file does  
✅ How the system starts and executes  
✅ Where to look to modify features  
✅ What can/cannot be safely changed  
✅ How to navigate the codebase confidently  

### Next Steps:
1. **Run the system:** `python main.py`
2. **Try live detection:** `python live_detection.py`
3. **Modify config:** Edit `config/config.json`
4. **Add features:** Follow the architecture patterns
5. **Train model:** Use `scripts/train.py`

---

**End of Complete Architectural Audit**

*Generated: 2026-02-01*  
*Project: Real-time Pothole Detection System v1.0.0*  
*Status: Production Ready ✅*
