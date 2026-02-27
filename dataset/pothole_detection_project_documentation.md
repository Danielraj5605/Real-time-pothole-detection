# An Integrated Multi-Sensor Embedded System for Real-Time Pothole Detection, Classification, and Driver Alerting

## Project Overview

This project presents a low-cost, embedded, multi-sensor system for real-time pothole detection, severity classification, and driver alerting using a Raspberry Pi platform. The system integrates camera-based computer vision, accelerometer-based vibration sensing, GPS-based geo-tagging, and a local alert server to provide a practical and scalable prototype for intelligent road monitoring and driver safety.

The primary goal is to overcome the limitations of single-sensor pothole detection systems by fusing vision and vibration data, enabling robust detection under varying environmental conditions and providing severity-aware, location-based alerts to drivers.

---

## Problem Statement

Road potholes are a major cause of vehicle damage, traffic delays, and accidents worldwide. Existing pothole detection systems primarily rely on single-sensor approaches, such as smartphone accelerometers or camera-based vision systems. These methods suffer from significant limitations:

- **Environmental sensitivity:** Vision-based systems fail under rain, low visibility, or poor lighting conditions, while accelerometer-only systems generate false positives due to speed bumps, rough roads, or vehicle-specific vibrations.
- **Lack of severity classification:** Most systems only detect pothole presence without estimating its depth or potential risk.
- **Delayed or offline reporting:** Many platforms rely on post-collection reporting rather than real-time driver alerts.
- **Fragmented implementation:** Existing research addresses individual aspects such as GPS tagging, sensor detection, or image processing, but lacks a fully integrated, low-cost, embedded system that combines all these features.

This project addresses these limitations by developing a fully integrated, multi-sensor embedded system that fuses camera and accelerometer data, classifies pothole severity, geo-tags potholes using GPS, and provides real-time alerts to nearby drivers using a local processing and alerting framework.

---

## Project Objectives

- **Accurate pothole detection under varying conditions**
  - Detect potholes reliably in different lighting and weather conditions.

- **Multimodal sensor fusion**
  - Fuse camera vision and MPU6050 accelerometer data to improve robustness and reduce false positives.

- **Severity classification**
  - Classify potholes into Low, Medium, and High severity based on vibration amplitude and visual features.

- **GPS-based geo-tagging**
  - Record latitude and longitude of detected potholes for mapping and alert purposes.

- **Real-time driver alerts**
  - Notify approaching vehicles of high-severity potholes using proximity-based alert logic.

- **Embedded prototype demonstration**
  - Implement the complete system on a Raspberry Pi to demonstrate feasibility, low cost, and real-time operation.

---

## Selected System Architecture

### Architecture Name:

**Hierarchical Multimodal Edge Fusion Architecture**

### Rationale

This architecture is selected as the best fit for the current local demo prototype and future deployment because it:

- Supports real-time, on-device processing
- Enables robust multimodal sensor fusion
- Reduces false positives through hierarchical decision logic
- Supports severity classification
- Integrates GPS-based geo-tagging
- Enables local, real-time driver alerts
- Scales easily to cloud or smart city platforms in future

---

## High-Level System Architecture

The system adopts a hierarchical, edge-based multimodal processing pipeline, where vibration sensing and vision-based detection are combined to confirm potholes, classify severity, and generate GPS-tagged alerts.

### Functional Block Flow

Camera Module
→ OpenCV + YOLOv5 (Vision CNN)
→ Visual pothole features and confidence

MPU6050 Accelerometer
→ Signal processing (filtering, peak detection, RMS)
→ Vibration features

Vision Features + Vibration Features
→ Multimodal Fusion and Severity Classification
→ Low / Medium / High Severity

Confirmed Pothole Event
→ GPS Module (NEO-6M)
→ Latitude and Longitude

Event Data
→ Local Storage (SQLite / JSON)

Stored Events + Current GPS
→ Flask / FastAPI Local Server
→ Proximity Check
→ Driver Alert (Buzzer / LED / Console)

---

## Hardware Components

- Raspberry Pi 4 (or equivalent) – Main processing unit
- Camera Module – Visual pothole detection
- MPU6050 Accelerometer/Gyroscope – Vibration sensing
- GPS Module (e.g., NEO-6M) – Geo-location
- Optional Buzzer / LED – Local alert indication

---

## Software Components

- Python – Main programming language
- OpenCV – Image preprocessing and handling
- YOLOv5 – CNN-based pothole detection
- Sensor Libraries (smbus, pyserial) – MPU6050 and GPS interfacing
- Flask / FastAPI – Local alert and data server
- SQLite / JSON – Lightweight local database

---

## Detailed Subsystem Description

### Vision Subsystem (YOLOv5 + OpenCV)

- Captures frames from camera module
- Applies preprocessing (resize, brightness/contrast, noise handling)
- Uses YOLOv5 to detect potholes
- Extracts:
  - Bounding box
  - Detection confidence
  - Visual size/area features

This subsystem handles environmental sensitivity and enables visual confirmation of potholes.

---

### Accelerometer Subsystem (MPU6050)

- Continuously monitors vibration data
- Applies:
  - Low-pass filtering
  - Peak detection
  - RMS / vibration energy computation

This subsystem helps distinguish potholes from speed bumps, rough roads, and general vibrations, reducing false positives.

---

### Multimodal Fusion and Severity Classification

- Combines visual and vibration features
- Uses hybrid fusion logic (feature-based + lightweight ML or rules)
- Inputs include:
  - Acceleration peak magnitude
  - Vibration duration
  - YOLO confidence score
  - Bounding box size

Outputs severity class:
- Low Severity
- Medium Severity
- High Severity

This enables depth/risk approximation and prioritization of dangerous potholes.

---

### GPS Geo-Tagging Subsystem

- Reads GPS coordinates when pothole is confirmed
- Tags each pothole event with:
  - Latitude
  - Longitude
  - Timestamp
  - Severity

This enables mapping, future integration, and proximity-based alerts.

---

### Local Storage Subsystem

- Uses SQLite or JSON to store pothole events
- Stored fields:
  - pothole_id
  - latitude
  - longitude
  - severity
  - timestamp

This supports offline operation and embedded deployment.

---

### Local Alert and Proximity Server

- Implemented using Flask or FastAPI
- Continuously monitors current GPS position
- Computes distance to stored potholes
- Triggers alert if:
  - Severity is High
  - Distance is below threshold

Alerts are generated using:
- Buzzer
- LED
- Console or UI message

---

## System Workflow

1. Mount camera, accelerometer, and GPS on test vehicle
2. Synchronize sensor data collection
3. Detect vibration spikes using MPU6050
4. Confirm potholes using YOLO-based vision processing
5. Fuse sensor data for final decision
6. Classify pothole severity (Low/Medium/High)
7. Geo-tag pothole using GPS
8. Store pothole data in local database
9. Run proximity checks for approaching vehicles
10. Trigger real-time alerts for high-severity potholes

---

## Testing and Validation

- Conduct controlled driving tests on varied road conditions
- Collect ground truth data for known potholes
- Compare detected potholes with ground truth
- Evaluate:
  - Detection accuracy
  - False positive rate
  - Severity classification correctness
  - Alert response reliability

---

## Expected Outcomes

- Functional embedded multi-sensor pothole detection system
- Improved detection accuracy under rain and low visibility
- Reliable severity classification
- Real-time driver alerts for high-risk potholes
- Low-cost, practical prototype
- Architecture suitable for future cloud and smart city integration

---

## Future Scope

- Cloud-based centralized pothole database
- Vehicle-to-vehicle alert sharing
- Deep learning-based sensor fusion
- Smart city dashboard integration
- Real-time map-based pothole visualization

---

## Final Architecture Summary

**Selected Architecture:** Hierarchical Multimodal Edge Fusion Architecture

This architecture fully satisfies all project requirements and provides a robust, scalable, and embedded-friendly foundation for real-time pothole detection, severity classification, GPS geo-tagging, and driver alerting.

