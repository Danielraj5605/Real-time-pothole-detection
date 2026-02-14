# Mock Data Generation and Testing Guide
## Real-Time Multimodal Pothole Detection System

---

## 📋 PROJECT CONCEPT - MY UNDERSTANDING

Based on my analysis of your codebase, here's what I understand about your project:

### **Core Concept**
Your system is a **multimodal real-time pothole detection and alert system** that combines:
1. **Vision (Camera)** - Captures video feed and detects potholes using YOLOv8
2. **Accelerometer** - Measures vehicle vibrations/impacts when driving over potholes
3. **GPS** - Records location (latitude/longitude) of detected potholes
4. **Fusion Logic** - Combines vision + accelerometer data to classify severity
5. **Alert System** - Warns nearby vehicles approaching known pothole locations

### **The Complete Workflow**

#### **Phase 1: Detection (Vehicle A - Data Collection)**
```
Vehicle A drives on road
    ↓
Camera captures video → YOLOv8 detects pothole visually
    ↓
Accelerometer detects impact/vibration
    ↓
Fusion System combines:
    - Vision confidence (0-100%)
    - Bbox area (pothole size)
    - Accelerometer peak (G-force)
    - Accelerometer RMS (vibration intensity)
    ↓
Severity Classification:
    - LOW: Small pothole, minor impact
    - MEDIUM: Moderate size, noticeable impact
    - HIGH: Large pothole, severe impact
    ↓
GPS captures location (lat, long)
    ↓
Data stored in database:
    {
        timestamp,
        latitude,
        longitude,
        severity: "LOW/MEDIUM/HIGH",
        vision_confidence,
        accel_peak,
        accel_rms
    }
```

#### **Phase 2: Alert (Vehicle B - Approaching Pothole)**
```
Vehicle B drives nearby
    ↓
GPS tracks current location
    ↓
System checks database for potholes within radius (e.g., 200m)
    ↓
If pothole found nearby:
    ↓
Calculate distance to pothole
    ↓
Trigger alert based on severity:
    - HIGH: Alert at 100m distance
    - MEDIUM: Alert at 50m distance
    - LOW: Alert at 20m distance
    ↓
Alert channels:
    - Console message
    - Buzzer (optional)
    - LED (optional)
    - Text-to-speech (optional)
```

### **Key Components**

1. **Vision System** (`pothole_detector.py`)
   - YOLOv8 model for pothole detection
   - 6-stage pipeline: Clean → Find → Track → Isolate → Read → Identify
   - Outputs: bounding box, confidence, area ratio

2. **Accelerometer System** (from config)
   - Samples at 50 Hz
   - Extracts features: peak acceleration, RMS vibration, peak-to-peak
   - Classifies severity using Random Forest

3. **GPS System** (from config)
   - Tracks vehicle location
   - Can run in simulation mode with jitter
   - Records lat/long at detection time

4. **Fusion System** (from config)
   - Rule-based or ML-based fusion
   - Combines vision weight (60%) + accel weight (40%)
   - Final severity classification

5. **Database** (SQLite)
   - Stores pothole detections
   - Enables proximity-based queries

---

## 🎯 MOCK DATA GENERATION STRATEGY

Since you don't have a physical vehicle setup, here's how to generate realistic mock data:

### **Option 1: Use Existing Dataset (Recommended for Quick Testing)**

Your project already has accelerometer data in `Datasets/Pothole/`:
- `trip1_sensors.csv` - 2219 rows of sensor data
- `trip1_potholes.csv` - 13 pothole timestamps
- Similar files for trips 2-5

**Data Format:**
```csv
timestamp,latitude,longitude,speed,accelerometerX,accelerometerY,accelerometerZ,gyroX,gyroY,gyroZ
1492638964.5,40.447444787735,-79.9441886564716,0.0,0.016998291015625,-0.962234497070312,0.203887939453125,...
```

### **Option 2: Generate Synthetic Mock Data**

I'll create a Python script that generates realistic mock data for testing.

---

## 🔧 MOCK DATA GENERATION SCRIPTS

### **Script 1: Generate Mock Trip Data**

```python
# scripts/generate_mock_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json

class MockDataGenerator:
    """Generate realistic mock data for pothole detection testing"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_trip(self, 
                     duration_minutes=10,
                     num_potholes=5,
                     sample_rate_hz=50,
                     start_lat=40.4474,
                     start_lon=-79.9441):
        """
        Generate a complete trip with sensor data and potholes
        
        Args:
            duration_minutes: Trip duration
            num_potholes: Number of potholes to simulate
            sample_rate_hz: Accelerometer sampling rate
            start_lat: Starting latitude
            start_lon: Starting longitude
        """
        
        # Calculate total samples
        total_samples = duration_minutes * 60 * sample_rate_hz
        
        # Generate timestamps
        start_time = datetime.now().timestamp()
        timestamps = [start_time + i/sample_rate_hz for i in range(total_samples)]
        
        # Generate GPS trajectory (simulate vehicle movement)
        latitudes, longitudes, speeds = self._generate_gps_trajectory(
            total_samples, start_lat, start_lon
        )
        
        # Generate baseline accelerometer data (normal driving)
        accel_x, accel_y, accel_z = self._generate_baseline_accel(total_samples)
        gyro_x, gyro_y, gyro_z = self._generate_baseline_gyro(total_samples)
        
        # Insert pothole events
        pothole_timestamps = []
        pothole_locations = []
        pothole_severities = []
        
        for i in range(num_potholes):
            # Random pothole location in trip
            pothole_idx = random.randint(
                int(total_samples * 0.1),  # Not at start
                int(total_samples * 0.9)   # Not at end
            )
            
            pothole_time = timestamps[pothole_idx]
            pothole_lat = latitudes[pothole_idx]
            pothole_lon = longitudes[pothole_idx]
            
            # Random severity
            severity = random.choice(['LOW', 'MEDIUM', 'HIGH'])
            
            # Inject pothole signature into accelerometer data
            accel_x, accel_y, accel_z = self._inject_pothole_signature(
                accel_x, accel_y, accel_z, pothole_idx, severity
            )
            
            pothole_timestamps.append(pothole_time)
            pothole_locations.append((pothole_lat, pothole_lon))
            pothole_severities.append(severity)
        
        # Create sensor dataframe
        sensor_df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude': latitudes,
            'longitude': longitudes,
            'speed': speeds,
            'accelerometerX': accel_x,
            'accelerometerY': accel_y,
            'accelerometerZ': accel_z,
            'gyroX': gyro_x,
            'gyroY': gyro_y,
            'gyroZ': gyro_z
        })
        
        # Create pothole dataframe
        pothole_df = pd.DataFrame({
            'timestamp': pothole_timestamps,
            'latitude': [loc[0] for loc in pothole_locations],
            'longitude': [loc[1] for loc in pothole_locations],
            'severity': pothole_severities
        })
        
        return sensor_df, pothole_df
    
    def _generate_gps_trajectory(self, num_samples, start_lat, start_lon):
        """Generate realistic GPS trajectory"""
        
        # Simulate vehicle moving at varying speeds
        speeds = []
        current_speed = 0.0
        
        for i in range(num_samples):
            # Gradual acceleration/deceleration
            if i < num_samples * 0.1:  # Accelerating
                current_speed = min(current_speed + 0.01, 15.0)
            elif i > num_samples * 0.9:  # Decelerating
                current_speed = max(current_speed - 0.01, 0.0)
            else:  # Cruising with variations
                current_speed += np.random.normal(0, 0.5)
                current_speed = np.clip(current_speed, 5.0, 15.0)
            
            speeds.append(current_speed)
        
        # Convert speed to GPS displacement
        # Approximate: 1 degree ≈ 111 km
        latitudes = [start_lat]
        longitudes = [start_lon]
        
        for i in range(1, num_samples):
            # Speed in m/s, sample rate in Hz
            distance_m = speeds[i] / 50.0  # meters per sample
            
            # Random direction (mostly forward with slight variations)
            angle = np.random.normal(0, 0.1)  # radians
            
            dlat = (distance_m * np.cos(angle)) / 111000  # degrees
            dlon = (distance_m * np.sin(angle)) / (111000 * np.cos(np.radians(latitudes[-1])))
            
            latitudes.append(latitudes[-1] + dlat)
            longitudes.append(longitudes[-1] + dlon)
        
        return latitudes, longitudes, speeds
    
    def _generate_baseline_accel(self, num_samples):
        """Generate baseline accelerometer data (normal driving)"""
        
        # Normal driving vibrations
        accel_x = np.random.normal(0.05, 0.03, num_samples)
        accel_y = np.random.normal(-0.96, 0.05, num_samples)  # Gravity
        accel_z = np.random.normal(0.20, 0.05, num_samples)
        
        # Add road texture noise
        accel_x += np.random.normal(0, 0.01, num_samples)
        accel_y += np.random.normal(0, 0.01, num_samples)
        accel_z += np.random.normal(0, 0.01, num_samples)
        
        return accel_x, accel_y, accel_z
    
    def _generate_baseline_gyro(self, num_samples):
        """Generate baseline gyroscope data"""
        
        gyro_x = np.random.normal(0, 0.02, num_samples)
        gyro_y = np.random.normal(0, 0.02, num_samples)
        gyro_z = np.random.normal(0, 0.01, num_samples)
        
        return gyro_x, gyro_y, gyro_z
    
    def _inject_pothole_signature(self, accel_x, accel_y, accel_z, 
                                  pothole_idx, severity):
        """Inject pothole impact signature into accelerometer data"""
        
        # Pothole impact parameters based on severity
        impact_params = {
            'LOW': {'peak': 0.5, 'duration': 10, 'rms': 0.15},
            'MEDIUM': {'peak': 1.5, 'duration': 15, 'rms': 0.5},
            'HIGH': {'peak': 3.0, 'duration': 20, 'rms': 1.0}
        }
        
        params = impact_params[severity]
        
        # Create impact signature (sharp spike + decay)
        duration = params['duration']
        half_dur = duration // 2
        
        for i in range(duration):
            idx = pothole_idx + i - half_dur
            if 0 <= idx < len(accel_x):
                # Gaussian-like impact
                t = (i - half_dur) / half_dur
                impact = params['peak'] * np.exp(-t**2)
                
                accel_y[idx] += impact * np.random.uniform(0.8, 1.2)
                accel_z[idx] += impact * 0.5 * np.random.uniform(0.8, 1.2)
                accel_x[idx] += impact * 0.3 * np.random.uniform(-1, 1)
        
        return accel_x, accel_y, accel_z


# Usage example
if __name__ == "__main__":
    generator = MockDataGenerator()
    
    # Generate 5 trips
    for trip_num in range(1, 6):
        print(f"Generating trip {trip_num}...")
        
        sensor_df, pothole_df = generator.generate_trip(
            duration_minutes=random.randint(5, 15),
            num_potholes=random.randint(3, 8),
            sample_rate_hz=50
        )
        
        # Save to CSV
        sensor_df.to_csv(f'Datasets/Pothole/mock_trip{trip_num}_sensors.csv', index=False)
        pothole_df.to_csv(f'Datasets/Pothole/mock_trip{trip_num}_potholes.csv', index=False)
        
        print(f"  - Sensor samples: {len(sensor_df)}")
        print(f"  - Potholes: {len(pothole_df)}")
        print(f"  - Severities: {pothole_df['severity'].value_counts().to_dict()}")
    
    print("\n✅ Mock data generation complete!")
```

### **Script 2: Generate Mock Video Frames with Potholes**

```python
# scripts/generate_mock_video.py
import cv2
import numpy as np
from pathlib import Path

class MockVideoGenerator:
    """Generate mock video frames with simulated potholes"""
    
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
    
    def generate_road_frame(self, has_pothole=False, pothole_severity='MEDIUM'):
        """Generate a single frame of road with optional pothole"""
        
        # Create base road texture
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 80
        
        # Add asphalt texture
        noise = np.random.randint(-20, 20, (self.height, self.width, 3))
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        
        # Add road markings
        cv2.line(frame, (self.width//2 - 5, 0), 
                (self.width//2 - 5, self.height), (255, 255, 255), 2)
        cv2.line(frame, (self.width//2 + 5, 0), 
                (self.width//2 + 5, self.height), (255, 255, 255), 2)
        
        if has_pothole:
            # Pothole parameters based on severity
            pothole_params = {
                'LOW': {'size': (40, 30), 'darkness': 40},
                'MEDIUM': {'size': (80, 60), 'darkness': 60},
                'HIGH': {'size': (120, 90), 'darkness': 80}
            }
            
            params = pothole_params[pothole_severity]
            
            # Random pothole location
            center_x = np.random.randint(100, self.width - 100)
            center_y = np.random.randint(200, self.height - 100)
            
            # Draw pothole (dark irregular shape)
            w, h = params['size']
            
            # Create elliptical pothole
            overlay = frame.copy()
            cv2.ellipse(overlay, (center_x, center_y), (w//2, h//2), 
                       0, 0, 360, (40, 40, 40), -1)
            
            # Add cracks around pothole
            for _ in range(5):
                angle = np.random.uniform(0, 2*np.pi)
                length = np.random.randint(20, 50)
                end_x = int(center_x + length * np.cos(angle))
                end_y = int(center_y + length * np.sin(angle))
                cv2.line(overlay, (center_x, center_y), (end_x, end_y), 
                        (30, 30, 30), 2)
            
            # Blend
            frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
            
            # Return frame and bounding box
            bbox = (center_x - w//2, center_y - h//2, 
                   center_x + w//2, center_y + h//2)
            return frame, bbox
        
        return frame, None
    
    def generate_video_sequence(self, output_path, duration_sec=10, fps=30,
                               pothole_frames=None):
        """
        Generate a video sequence
        
        Args:
            output_path: Path to save video
            duration_sec: Video duration in seconds
            fps: Frames per second
            pothole_frames: List of (frame_num, severity) tuples
        """
        
        total_frames = duration_sec * fps
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                             (self.width, self.height))
        
        pothole_dict = {}
        if pothole_frames:
            pothole_dict = {frame_num: severity 
                          for frame_num, severity in pothole_frames}
        
        annotations = []
        
        for frame_num in range(total_frames):
            has_pothole = frame_num in pothole_dict
            severity = pothole_dict.get(frame_num, 'MEDIUM')
            
            frame, bbox = self.generate_road_frame(has_pothole, severity)
            
            if bbox:
                annotations.append({
                    'frame': frame_num,
                    'bbox': bbox,
                    'severity': severity
                })
            
            out.write(frame)
        
        out.release()
        
        return annotations


# Usage
if __name__ == "__main__":
    generator = MockVideoGenerator()
    
    # Generate test video with potholes at specific frames
    pothole_frames = [
        (90, 'LOW'),
        (180, 'MEDIUM'),
        (270, 'HIGH'),
        (360, 'MEDIUM'),
        (450, 'LOW')
    ]
    
    annotations = generator.generate_video_sequence(
        'Datasets/mock_video.mp4',
        duration_sec=20,
        fps=30,
        pothole_frames=pothole_frames
    )
    
    print(f"✅ Generated video with {len(annotations)} potholes")
    for ann in annotations:
        print(f"  Frame {ann['frame']}: {ann['severity']} at {ann['bbox']}")
```

---

## 🧪 TESTING STRATEGY

### **Test 1: Vision System Only**

Test the YOLOv8 detection on mock images:

```bash
# Generate mock images
python scripts/generate_mock_video.py

# Test detection
python pothole_detector.py --model models/weights/pothole_pretrained_95percent.pt --test
```

### **Test 2: Accelerometer Processing**

Test accelerometer feature extraction:

```python
# scripts/test_accelerometer.py
import pandas as pd
import numpy as np

def extract_accel_features(sensor_df, window_size=50):
    """Extract accelerometer features from sensor data"""
    
    features = []
    
    for i in range(0, len(sensor_df) - window_size, window_size//2):
        window = sensor_df.iloc[i:i+window_size]
        
        # Calculate features
        accel_y = window['accelerometerY'].values
        
        feature = {
            'timestamp': window['timestamp'].iloc[window_size//2],
            'peak_accel': np.max(np.abs(accel_y)),
            'rms_accel': np.sqrt(np.mean(accel_y**2)),
            'peak_to_peak': np.max(accel_y) - np.min(accel_y),
            'mean_accel': np.mean(accel_y),
            'std_accel': np.std(accel_y)
        }
        
        features.append(feature)
    
    return pd.DataFrame(features)

# Test
sensor_df = pd.read_csv('Datasets/Pothole/trip1_sensors.csv')
features_df = extract_accel_features(sensor_df)
print(features_df.head())
```

### **Test 3: Fusion System**

Test combining vision + accelerometer:

```python
# scripts/test_fusion.py

def fusion_classify_severity(vision_conf, bbox_area_ratio, 
                             accel_peak, accel_rms,
                             vision_weight=0.6, accel_weight=0.4):
    """
    Classify pothole severity using fusion
    
    Args:
        vision_conf: Vision confidence (0-1)
        bbox_area_ratio: Bounding box area / frame area
        accel_peak: Peak acceleration (G)
        accel_rms: RMS acceleration (G)
    """
    
    # Vision score
    vision_score = vision_conf * bbox_area_ratio
    
    # Accelerometer score
    accel_score = min(accel_peak / 3.0, 1.0)  # Normalize to 0-1
    
    # Weighted fusion
    combined_score = vision_weight * vision_score + accel_weight * accel_score
    
    # Classify
    if combined_score > 0.7 or accel_peak > 2.0:
        return 'HIGH'
    elif combined_score > 0.4 or accel_peak > 1.0:
        return 'MEDIUM'
    else:
        return 'LOW'

# Test cases
test_cases = [
    # (vision_conf, bbox_area, accel_peak, accel_rms, expected)
    (0.9, 0.15, 2.5, 0.8, 'HIGH'),
    (0.7, 0.08, 1.2, 0.4, 'MEDIUM'),
    (0.5, 0.03, 0.4, 0.1, 'LOW'),
]

for vision_conf, bbox_area, accel_peak, accel_rms, expected in test_cases:
    result = fusion_classify_severity(vision_conf, bbox_area, accel_peak, accel_rms)
    status = "✅" if result == expected else "❌"
    print(f"{status} Vision:{vision_conf:.1f} Area:{bbox_area:.2f} "
          f"Peak:{accel_peak:.1f}G → {result} (expected {expected})")
```

### **Test 4: Alert System**

Test proximity-based alerts:

```python
# scripts/test_alerts.py
import math

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates (Haversine formula)"""
    
    R = 6371000  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def check_proximity_alert(current_lat, current_lon, pothole_db, max_distance=200):
    """
    Check if vehicle is approaching any known potholes
    
    Args:
        current_lat, current_lon: Current vehicle position
        pothole_db: List of dicts with pothole data
        max_distance: Maximum alert distance (meters)
    """
    
    alerts = []
    
    for pothole in pothole_db:
        distance = calculate_distance(
            current_lat, current_lon,
            pothole['latitude'], pothole['longitude']
        )
        
        # Alert thresholds based on severity
        alert_thresholds = {
            'HIGH': 100,
            'MEDIUM': 50,
            'LOW': 20
        }
        
        threshold = alert_thresholds.get(pothole['severity'], 50)
        
        if distance <= threshold:
            alerts.append({
                'pothole': pothole,
                'distance': distance,
                'message': f"⚠️ {pothole['severity']} pothole ahead in {distance:.0f}m!"
            })
    
    return alerts

# Test
pothole_db = [
    {'latitude': 40.4474, 'longitude': -79.9442, 'severity': 'HIGH'},
    {'latitude': 40.4475, 'longitude': -79.9443, 'severity': 'MEDIUM'},
    {'latitude': 40.4476, 'longitude': -79.9444, 'severity': 'LOW'},
]

# Simulate vehicle approaching
current_lat, current_lon = 40.4473, -79.9441

alerts = check_proximity_alert(current_lat, current_lon, pothole_db)

for alert in alerts:
    print(alert['message'])
```

---

## 📊 COMPLETE END-TO-END TEST

```python
# scripts/end_to_end_test.py
"""
Complete end-to-end test of the pothole detection system
"""

import pandas as pd
import numpy as np
from pathlib import Path

def run_end_to_end_test():
    """Run complete system test"""
    
    print("=" * 60)
    print("  POTHOLE DETECTION SYSTEM - END-TO-END TEST")
    print("=" * 60)
    
    # Step 1: Load mock data
    print("\n[1/5] Loading mock sensor data...")
    sensor_df = pd.read_csv('Datasets/Pothole/trip1_sensors.csv')
    pothole_df = pd.read_csv('Datasets/Pothole/trip1_potholes.csv')
    print(f"  ✅ Loaded {len(sensor_df)} sensor samples")
    print(f"  ✅ Loaded {len(pothole_df)} pothole events")
    
    # Step 2: Process accelerometer data
    print("\n[2/5] Processing accelerometer data...")
    # Extract features at pothole timestamps
    detections = []
    for _, pothole in pothole_df.iterrows():
        timestamp = pothole['timestamp']
        
        # Find sensor data around this timestamp
        mask = (sensor_df['timestamp'] >= timestamp - 0.5) & \
               (sensor_df['timestamp'] <= timestamp + 0.5)
        window = sensor_df[mask]
        
        if len(window) > 0:
            accel_y = window['accelerometerY'].values
            accel_peak = np.max(np.abs(accel_y))
            accel_rms = np.sqrt(np.mean(accel_y**2))
            
            detections.append({
                'timestamp': timestamp,
                'accel_peak': accel_peak,
                'accel_rms': accel_rms
            })
    
    print(f"  ✅ Processed {len(detections)} pothole events")
    
    # Step 3: Simulate vision detection
    print("\n[3/5] Simulating vision detection...")
    for detection in detections:
        # Simulate YOLOv8 detection
        detection['vision_conf'] = np.random.uniform(0.7, 0.95)
        detection['bbox_area_ratio'] = np.random.uniform(0.05, 0.15)
    
    print(f"  ✅ Generated vision data for {len(detections)} detections")
    
    # Step 4: Fusion and severity classification
    print("\n[4/5] Classifying severity...")
    for detection in detections:
        # Simple fusion logic
        vision_score = detection['vision_conf'] * detection['bbox_area_ratio']
        accel_score = min(detection['accel_peak'] / 3.0, 1.0)
        combined = 0.6 * vision_score + 0.4 * accel_score
        
        if combined > 0.7 or detection['accel_peak'] > 2.0:
            severity = 'HIGH'
        elif combined > 0.4 or detection['accel_peak'] > 1.0:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        detection['severity'] = severity
    
    severity_counts = pd.Series([d['severity'] for d in detections]).value_counts()
    print(f"  ✅ Severity distribution: {severity_counts.to_dict()}")
    
    # Step 5: Simulate alert system
    print("\n[5/5] Testing alert system...")
    # Simulate vehicle approaching first pothole
    if len(detections) > 0:
        first_pothole = detections[0]
        print(f"  ⚠️ ALERT: {first_pothole['severity']} pothole detected!")
        print(f"     - Vision confidence: {first_pothole['vision_conf']:.1%}")
        print(f"     - Accel peak: {first_pothole['accel_peak']:.2f}G")
        print(f"     - Accel RMS: {first_pothole['accel_rms']:.2f}G")
    
    print("\n" + "=" * 60)
    print("  ✅ END-TO-END TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    run_end_to_end_test()
```

---

## 🚀 QUICK START TESTING

### **Step 1: Generate Mock Data**
```bash
python scripts/generate_mock_data.py
```

### **Step 2: Test Vision System**
```bash
python pothole_detector.py --model models/weights/pothole_pretrained_95percent.pt --test
```

### **Step 3: Run End-to-End Test**
```bash
python scripts/end_to_end_test.py
```

### **Step 4: Test with Existing Data**
```bash
# Use the existing trip data
python scripts/test_accelerometer.py
python scripts/test_fusion.py
python scripts/test_alerts.py
```

---

## 📝 SUMMARY

Your project is a **comprehensive multimodal pothole detection and alert system** that:

1. **Detects** potholes using camera + accelerometer
2. **Classifies** severity (LOW/MEDIUM/HIGH) using data fusion
3. **Records** location with GPS
4. **Alerts** nearby vehicles approaching known potholes

The mock data generation scripts I've provided will help you:
- Generate realistic sensor data
- Test each component independently
- Run end-to-end system tests
- Validate the fusion logic
- Test the alert system

All without needing physical hardware!

---

**Next Steps:**
1. Create the mock data generation scripts
2. Run the tests
3. Validate the system works as expected
4. Deploy to real hardware when ready

Let me know if you need help with any specific part!
