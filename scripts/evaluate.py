#!/usr/bin/env python3
"""
Evaluation Script

Evaluates the trained pothole detection model on test data.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --weights models/weights/pothole_best.pt
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.utils import setup_logger
from src.vision import PotholeDetector, VisionFeatureExtractor
from src.accelerometer import AccelerometerProcessor, AccelFeatureExtractor, SeverityClassifier
from src.fusion import FusionEngine


def evaluate_vision(detector, test_images: list, output_dir: Path):
    """Evaluate vision pipeline on test images."""
    print("\n📷 Vision Pipeline Evaluation")
    print("-"*40)
    
    feature_extractor = VisionFeatureExtractor()
    results = []
    
    for img_path in test_images:
        detections = detector.detect(str(img_path))
        
        # Get image dimensions
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        features = feature_extractor.extract(detections, w, h)
        
        results.append({
            'path': str(img_path),
            'detected': features.detected,
            'confidence': features.confidence,
            'num_detections': features.num_detections
        })
    
    # Calculate metrics
    n_detected = sum(1 for r in results if r['detected'])
    avg_conf = np.mean([r['confidence'] for r in results if r['detected']]) if n_detected > 0 else 0
    
    print(f"   Images evaluated: {len(results)}")
    print(f"   Detections: {n_detected} ({n_detected/len(results)*100:.1f}%)")
    print(f"   Average confidence: {avg_conf:.2%}")
    
    return results


def evaluate_accelerometer(processor, classifier, csv_path: str):
    """Evaluate accelerometer pipeline."""
    print("\n📊 Accelerometer Pipeline Evaluation")
    print("-"*40)
    
    feature_extractor = AccelFeatureExtractor()
    
    windows = list(processor.process_file(csv_path))
    print(f"   Windows processed: {len(windows)}")
    
    # Classify all windows
    results = []
    severity_counts = {'none': 0, 'low': 0, 'medium': 0, 'high': 0}
    
    for window in windows:
        features = feature_extractor.extract(window)
        prediction = classifier.predict(features)
        
        severity_counts[prediction.severity] += 1
        results.append({
            'timestamp': window.start_time,
            'peak': features.peak_acceleration,
            'rms': features.rms_vibration,
            'severity': prediction.severity,
            'confidence': prediction.confidence
        })
    
    print(f"   Severity distribution:")
    for sev, count in severity_counts.items():
        pct = count / len(results) * 100 if results else 0
        print(f"      {sev}: {count} ({pct:.1f}%)")
    
    return results


def evaluate_fusion(vision_results, accel_results, fusion_engine: FusionEngine):
    """Evaluate fusion pipeline."""
    print("\n🔀 Fusion Pipeline Evaluation")
    print("-"*40)
    
    # For demo, we'll just show fusion works
    # In production, you'd have aligned pairs
    
    n_vision = sum(1 for r in vision_results if r['detected'])
    n_accel = sum(1 for r in accel_results if r['severity'] != 'none')
    
    print(f"   Vision detections: {n_vision}")
    print(f"   Accelerometer detections: {n_accel}")
    
    # Calculate potential fusion matches
    # This is simplified - real implementation would use time alignment
    fusion_potential = min(n_vision, n_accel)
    print(f"   Potential fusion matches: {fusion_potential}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pothole detection models"
    )
    parser.add_argument(
        '--weights', type=str, default=None,
        help='Path to YOLO weights'
    )
    parser.add_argument(
        '--images', type=str, default='Datasets/Pothole_Image_Data',
        help='Test images directory'
    )
    parser.add_argument(
        '--accel', type=str, default='Datasets/Pothole/trip1_sensors.csv',
        help='Test accelerometer data'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger("evaluation", level="INFO")
    
    print("="*60)
    print("📊 POTHOLE DETECTION EVALUATION")
    print("="*60)
    
    # Initialize models
    print("\n🔧 Initializing models...")
    
    weights = args.weights or "yolov8n.pt"
    detector = PotholeDetector(
        model_path=weights,
        confidence_threshold=0.25
    )
    
    processor = AccelerometerProcessor(
        window_size=50,
        overlap_ratio=0.5
    )
    
    classifier = SeverityClassifier()
    classifier.train_synthetic(n_samples_per_class=200)
    
    fusion_engine = FusionEngine()
    
    # Find test images
    images_dir = project_root / args.images
    test_images = list(images_dir.glob("*.jpg"))[:50]  # Limit for demo
    
    print(f"\n📂 Test images: {len(test_images)}")
    
    # Evaluate vision
    output_dir = project_root / "results/evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vision_results = evaluate_vision(detector, test_images, output_dir)
    
    # Evaluate accelerometer
    accel_csv = project_root / args.accel
    if accel_csv.exists():
        accel_results = evaluate_accelerometer(processor, classifier, str(accel_csv))
    else:
        accel_results = []
        print(f"\n⚠️  Accelerometer data not found: {accel_csv}")
    
    # Evaluate fusion
    if vision_results and accel_results:
        evaluate_fusion(vision_results, accel_results, fusion_engine)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
