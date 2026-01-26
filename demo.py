#!/usr/bin/env python3
"""
Multimodal Pothole Detection Demo

Demonstrates the complete pothole detection pipeline:
1. Vision-based detection using YOLOv8
2. Accelerometer-based severity classification
3. Multimodal fusion for robust detection

Usage:
    python demo.py                     # Run with sample data
    python demo.py --image path.jpg    # Test with single image
    python demo.py --accel trip.csv    # Test with sensor data
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
from datetime import datetime

from src.utils import get_config, get_logger, setup_logger
from src.vision import PotholeDetector, VisionFeatureExtractor
from src.accelerometer import AccelerometerProcessor, AccelFeatureExtractor, SeverityClassifier
from src.fusion import FusionEngine, AlertManager


def print_banner():
    """Print demo banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║           MULTIMODAL POTHOLE DETECTION SYSTEM                   ║
║                                                                  ║
║   Vision (YOLOv8) + Accelerometer Fusion Pipeline               ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def demo_vision_only(
    detector: PotholeDetector,
    feature_extractor: VisionFeatureExtractor,
    image_path: str,
    output_dir: Path
):
    """Demonstrate vision-only detection."""
    print("\n" + "="*60)
    print("📷 VISION PIPELINE DEMO")
    print("="*60)
    
    # Load and detect
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None
    
    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")
    
    # Run detection
    print("\n🔍 Running pothole detection...")
    detections = detector.detect(image)
    
    print(f"\n📊 Detection Results:")
    print(f"   Potholes found: {len(detections)}")
    
    for i, det in enumerate(detections):
        print(f"\n   Detection {i+1}:")
        print(f"   - Confidence: {det.confidence:.2%}")
        print(f"   - Bounding Box: {det.bbox}")
        print(f"   - Area: {det.area:.0f} pixels²")
        print(f"   - Aspect Ratio: {det.aspect_ratio:.2f}")
    
    # Extract features
    features = feature_extractor.extract(detections, width, height)
    print(f"\n📈 Extracted Features:")
    print(f"   - Detected: {features.detected}")
    print(f"   - Best Confidence: {features.confidence:.2%}")
    print(f"   - Normalized Area: {features.bbox_area_normalized:.4f}")
    print(f"   - Num Detections: {features.num_detections}")
    print(f"   - Visual Severity Hint: {feature_extractor.compute_severity_hint(features)}")
    
    # Visualize
    vis_image = detector.visualize(image, detections)
    output_path = output_dir / "vision_detection.jpg"
    cv2.imwrite(str(output_path), vis_image)
    print(f"\n💾 Saved visualization: {output_path}")
    
    return features


def demo_accelerometer_only(
    processor: AccelerometerProcessor,
    feature_extractor: AccelFeatureExtractor,
    classifier: SeverityClassifier,
    csv_path: str,
    pothole_csv: str = None
):
    """Demonstrate accelerometer-only analysis."""
    print("\n" + "="*60)
    print("📊 ACCELEROMETER PIPELINE DEMO")
    print("="*60)
    
    # Load pothole timestamps if available
    pothole_timestamps = []
    if pothole_csv and Path(pothole_csv).exists():
        import pandas as pd
        df = pd.read_csv(pothole_csv)
        pothole_timestamps = df['timestamp'].tolist()
        print(f"\n📍 Loaded {len(pothole_timestamps)} known pothole timestamps")
    
    # Process sensor file
    print(f"\n📂 Processing: {csv_path}")
    
    windows = list(processor.process_file(csv_path, pothole_timestamps))
    print(f"   Generated {len(windows)} analysis windows")
    
    # Analyze first few windows
    print("\n🔬 Sample Window Analysis:")
    
    pothole_windows = []
    for i, window in enumerate(windows[:20]):  # First 20 windows
        features = feature_extractor.extract(window)
        is_candidate = feature_extractor.is_pothole_candidate(features)
        
        if is_candidate:
            prediction = classifier.predict(features)
            
            print(f"\n   Window {i+1} (t={window.start_time:.1f}s):")
            print(f"   - Peak: {features.peak_acceleration:.3f}g")
            print(f"   - RMS: {features.rms_vibration:.3f}g")
            print(f"   - Crest Factor: {features.crest_factor:.2f}")
            print(f"   - Severity: {prediction.severity} ({prediction.confidence:.0%})")
            
            if window.latitude and window.longitude:
                print(f"   - Location: ({window.latitude:.4f}, {window.longitude:.4f})")
            
            pothole_windows.append((window, features, prediction))
    
    print(f"\n📈 Summary:")
    print(f"   Total windows analyzed: {len(windows)}")
    print(f"   Pothole candidates: {len(pothole_windows)}")
    
    return pothole_windows


def demo_fusion(
    vision_features,
    accel_windows,
    fusion_engine: FusionEngine,
    alert_manager: AlertManager
):
    """Demonstrate multimodal fusion."""
    print("\n" + "="*60)
    print("🔀 MULTIMODAL FUSION DEMO")
    print("="*60)
    
    # Simulate fusion for pothole windows
    if not accel_windows:
        print("\n⚠️  No accelerometer data available for fusion")
        
        if vision_features:
            print("\n🔍 Vision-only fusion:")
            result = fusion_engine.fuse(vision_features, None)
            print(f"   {result}")
            
            if result.pothole_detected:
                alert = alert_manager.process(result)
                if alert:
                    print(f"   🚨 {alert}")
        return
    
    print(f"\n🔄 Fusing {len(accel_windows)} candidate windows...")
    
    for i, (window, accel_features, prediction) in enumerate(accel_windows[:5]):
        # Combine with vision (if available)
        result = fusion_engine.fuse(
            vision_features if i == 0 else None,  # Use vision for first only
            accel_features,
            latitude=window.latitude,
            longitude=window.longitude,
            accel_severity=prediction
        )
        
        print(f"\n   Fusion {i+1}:")
        print(f"   - {result}")
        
        # Process alert
        alert = alert_manager.process(result)
        if alert:
            print(f"   🚨 {alert}")
    
    # Show statistics
    stats = alert_manager.get_statistics()
    print(f"\n📊 Alert Statistics:")
    print(f"   Total alerts: {stats['total_alerts']}")
    print(f"   By severity: {stats['severity_counts']}")
    print(f"   Avg confidence: {stats['avg_confidence']:.0%}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Multimodal Pothole Detection Demo"
    )
    parser.add_argument(
        '--image', type=str,
        help='Path to test image'
    )
    parser.add_argument(
        '--accel', type=str,
        help='Path to accelerometer CSV file'
    )
    parser.add_argument(
        '--potholes', type=str,
        help='Path to pothole timestamps CSV'
    )
    parser.add_argument(
        '--config', type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', type=str,
        default='results/demo_outputs',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    print("⚙️  Loading configuration...")
    try:
        config = get_config(args.config)
        print(f"   Loaded: {args.config}")
    except FileNotFoundError:
        print(f"   ⚠️  Config not found, using defaults")
        config = None
    
    # Setup logger
    logger = setup_logger(
        "demo",
        level="INFO",
        log_file=str(output_dir / "demo.log")
    )
    
    # Get sample paths from config or use defaults
    project_root = Path(__file__).parent
    
    sample_image = args.image
    if not sample_image:
        # Try to find a sample image
        candidates = [
            project_root / "Datasets/images/vlcsnap-00001.jpg",
            project_root / "Datasets/Pothole_Image_Data/1.jpg"
        ]
        for path in candidates:
            if path.exists():
                sample_image = str(path)
                break
    
    sample_accel = args.accel
    if not sample_accel:
        default_accel = project_root / "Datasets/Pothole/trip1_sensors.csv"
        if default_accel.exists():
            sample_accel = str(default_accel)
    
    sample_potholes = args.potholes
    if not sample_potholes:
        default_potholes = project_root / "Datasets/Pothole/trip1_potholes.csv"
        if default_potholes.exists():
            sample_potholes = str(default_potholes)
    
    # Initialize pipelines
    print("\n🚀 Initializing pipelines...")
    
    # Vision pipeline
    print("   📷 Vision detector...")
    detector = PotholeDetector(
        model_path="yolov8n.pt",  # Will use pretrained if custom not found
        confidence_threshold=0.25
    )
    vision_extractor = VisionFeatureExtractor()
    
    # Accelerometer pipeline
    print("   📊 Accelerometer processor...")
    accel_processor = AccelerometerProcessor(
        window_size=50,
        overlap_ratio=0.5,
        sample_rate=50.0,
        apply_filter=True
    )
    accel_extractor = AccelFeatureExtractor()
    
    # Severity classifier
    print("   🎯 Severity classifier...")
    classifier = SeverityClassifier(model_type="random_forest")
    # Train on synthetic data for demo
    print("      Training on synthetic data...")
    classifier.train_synthetic(n_samples_per_class=200)
    
    # Fusion engine
    print("   🔀 Fusion engine...")
    fusion_engine = FusionEngine(
        method="rule_based",
        vision_weight=0.6,
        accel_weight=0.4
    )
    
    # Alert manager
    alert_manager = AlertManager(
        debounce_seconds=2.0,
        min_severity='low'
    )
    
    # Add console alert callback
    def print_alert(alert):
        severity_emoji = {'low': '⚠️', 'medium': '🔶', 'high': '🔴'}
        emoji = severity_emoji.get(alert.severity, '❓')
        print(f"\n{emoji} NEW ALERT: {alert}")
    
    alert_manager.add_callback(print_alert)
    
    print("\n✅ All pipelines initialized!")
    
    # Run demos
    vision_features = None
    accel_windows = None
    
    # Vision demo
    if sample_image:
        vision_features = demo_vision_only(
            detector, vision_extractor,
            sample_image, output_dir
        )
    else:
        print("\n⚠️  No sample image found for vision demo")
    
    # Accelerometer demo
    if sample_accel:
        accel_windows = demo_accelerometer_only(
            accel_processor, accel_extractor, classifier,
            sample_accel, sample_potholes
        )
    else:
        print("\n⚠️  No sample accelerometer data found")
    
    # Fusion demo
    demo_fusion(
        vision_features, accel_windows or [],
        fusion_engine, alert_manager
    )
    
    # Summary
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE")
    print("="*60)
    print(f"\n📁 Results saved to: {output_dir}")
    print("\n💡 Next steps:")
    print("   1. Train YOLOv8 on pothole dataset: python scripts/train.py")
    print("   2. Prepare dataset: python scripts/prepare_dataset.py")
    print("   3. Run evaluation: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
