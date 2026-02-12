#!/usr/bin/env python3
"""
Test Pre-Trained Pothole Detection Model

Tests the downloaded 95% accuracy model on your dataset.
"""

import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Setup
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "weights" / "pothole_pretrained_95percent.pt"
DATASET_PATHS = [
    PROJECT_ROOT / "Datasets" / "train" / "images",
    PROJECT_ROOT / "Datasets" / "val" / "images",
]
OUTPUT_DIR = PROJECT_ROOT / "results" / "pretrained_test_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*60)
print("TESTING PRE-TRAINED MODEL (95% ACCURACY)")
print("="*60)

# Check model exists
if not MODEL_PATH.exists():
    print(f"\n[ERROR] Model not found: {MODEL_PATH}")
    print("Please make sure the model is in the correct location.")
    sys.exit(1)

# Load model
print(f"\nLoading pre-trained model...")
print(f"Model: {MODEL_PATH.name}")
model = YOLO(str(MODEL_PATH))
print("[OK] Model loaded successfully!")

# Find images
print("\nFinding images in dataset...")
image_files = []
for dataset_path in DATASET_PATHS:
    if dataset_path.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(list(dataset_path.glob(ext)))

if not image_files:
    print("[ERROR] No images found!")
    sys.exit(1)

# Test on first 20 images
image_files = image_files[:20]
print(f"[OK] Testing on {len(image_files)} images")

# Run detection
print("\n" + "="*60)
print("RUNNING DETECTION")
print("="*60)

total_detections = 0
images_with_detections = 0
confidence_threshold = 0.25

for idx, img_path in enumerate(image_files, 1):
    # Run detection
    results = model(str(img_path), conf=confidence_threshold, verbose=False)
    
    # Count detections
    num_detections = 0
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            num_detections = len(result.boxes)
    
    if num_detections > 0:
        total_detections += num_detections
        images_with_detections += 1
        
        # Save annotated image
        annotated = results[0].plot()
        output_path = OUTPUT_DIR / f"{img_path.stem}_detected.jpg"
        cv2.imwrite(str(output_path), annotated)
        
        print(f"  [{idx:2d}/{len(image_files)}] {img_path.name}: {num_detections} potholes detected ✓")
        print(f"           Saved: {output_path.name}")
    else:
        print(f"  [{idx:2d}/{len(image_files)}] {img_path.name}: No potholes detected")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\nTotal images tested: {len(image_files)}")
print(f"Images with potholes: {images_with_detections}")
print(f"Total potholes detected: {total_detections}")
print(f"Detection rate: {images_with_detections/len(image_files)*100:.1f}%")

if images_with_detections > 0:
    print(f"\n✓ Annotated images saved to:")
    print(f"  {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"1. Check the saved images to see the detections")
    print(f"2. Run live detection: python pothole_detector.py --model {MODEL_PATH}")
else:
    print(f"\n[INFO] No potholes detected in these images.")
    print(f"This could mean:")
    print(f"  - The images don't contain visible potholes")
    print(f"  - The potholes are too small or unclear")
    print(f"  - Try testing with different images")

print("\n" + "="*60)
