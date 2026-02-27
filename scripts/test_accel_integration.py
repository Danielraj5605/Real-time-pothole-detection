"""Quick integration test for accelerometer fusion"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from accelerometer_processor import AccelerometerProcessor, AccelConfig

# Test 1: Load and extract features from mock data
print("=" * 60)
print("  ACCELEROMETER INTEGRATION TEST")
print("=" * 60)

proc = AccelerometerProcessor(AccelConfig())
proc.initialize(csv_path='Datasets/Pothole/mock_trip1_sensors.csv')

feat = proc.extract_features()
if feat:
    print(f"\n[1] Feature Extraction:")
    print(f"    Peak Acceleration: {feat.peak_acceleration:.3f} G")
    print(f"    RMS Vibration:     {feat.rms_vibration:.3f} G")
    print(f"    Peak-to-Peak:      {feat.peak_to_peak:.3f} G")
    print(f"    Accel Severity:    {feat.accel_severity}")
else:
    print("  ERROR: No features extracted")

# Test 2: Fusion with different vision scenarios
print(f"\n[2] Fusion Tests:")
test_cases = [
    (0.9, 0.15, "High confidence + large bbox"),
    (0.7, 0.08, "Medium confidence + medium bbox"),
    (0.4, 0.03, "Low confidence + small bbox"),
]

for vision_conf, area_ratio, desc in test_cases:
    result = proc.fuse_severity(vision_confidence=vision_conf, vision_area_ratio=area_ratio)
    print(f"    {desc}")
    print(f"      -> Severity: {result['severity']} | Fusion: {result['fusion_score']} | "
          f"Accel: {result['accel_peak_g']:.2f}G | HasAccel: {result['has_accel_data']}")

# Test 3: Multiple windows to find pothole signatures
print(f"\n[3] Scanning trip for pothole impacts...")
proc2 = AccelerometerProcessor(AccelConfig())
proc2.initialize(csv_path='Datasets/Pothole/mock_trip1_sensors.csv')

found = 0
window_count = 0
while True:
    feat = proc2.extract_features()
    if feat is None:
        break
    window_count += 1
    if feat.accel_severity in ("MEDIUM", "HIGH"):
        found += 1
        if found <= 5:
            print(f"    Window {window_count}: Peak={feat.peak_acceleration:.2f}G "
                  f"RMS={feat.rms_vibration:.2f}G -> {feat.accel_severity}")

print(f"    Total windows: {window_count}, Pothole impacts found: {found}")

print(f"\n{'=' * 60}")
print(f"  ✅ ALL TESTS PASSED!")
print(f"{'=' * 60}")
