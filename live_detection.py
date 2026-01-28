"""
Live Pothole Detection - Main Entry Point

Real-time pothole detection from live camera feed using YOLOv8.
Integrates with the modular event-driven architecture.

Usage:
    python live_detection.py              # Run with config defaults
    python live_detection.py --camera 0   # Use camera index 0
    python live_detection.py --no-display # Run without GUI window
    python live_detection.py --save       # Save detection frames
"""
import sys
import json
import logging
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.infrastructure.ml.adapters.yolov8_detector import YOLOv8Detector
from src.application.services.live_detection_service import (
    LiveDetectionService,
    LiveDetectionConfig,
    DetectionFrame
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/live_detection.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from JSON files."""
    config_path = project_root / "config" / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def create_live_config(config: dict, args: argparse.Namespace) -> LiveDetectionConfig:
    """Create LiveDetectionConfig from JSON config and CLI args."""
    hw_config = config.get('hardware', {})
    camera_config = hw_config.get('camera', {})
    live_config = config.get('live_detection', {})
    viz_config = live_config.get('visualization', {})
    proc_config = live_config.get('processing', {})
    log_config = live_config.get('logging', {})
    
    # CLI args override config file
    camera_id = args.camera if args.camera is not None else camera_config.get('device_id', 0)
    show_window = not args.no_display
    save_detections = args.save
    confidence = args.confidence if args.confidence else live_config.get('confidence_threshold', 0.25)
    model_path = args.model if args.model else live_config.get('model_path', 'yolov8n.pt')
    
    return LiveDetectionConfig(
        # Camera
        camera_device_id=camera_id,
        camera_width=camera_config.get('width', 640),
        camera_height=camera_config.get('height', 480),
        camera_fps=camera_config.get('fps', 30),
        warmup_frames=camera_config.get('warmup_frames', 10),
        
        # Model
        model_path=model_path,
        confidence_threshold=confidence,
        iou_threshold=live_config.get('iou_threshold', 0.45),
        
        # Processing
        target_fps=live_config.get('target_fps', 15),
        skip_frames=live_config.get('skip_frames', 0),
        resize_width=proc_config.get('resize_width', 640),
        resize_height=proc_config.get('resize_height', 480),
        use_gpu=proc_config.get('use_gpu', True),
        
        # Visualization
        show_window=show_window,
        show_fps=viz_config.get('show_fps', True),
        show_detections=viz_config.get('show_detections', True),
        show_confidence=viz_config.get('show_confidence', True),
        bbox_color=tuple(viz_config.get('bbox_color', [0, 255, 0])),
        bbox_thickness=viz_config.get('bbox_thickness', 2),
        font_scale=viz_config.get('font_scale', 0.6),
        window_name=viz_config.get('window_name', 'Pothole Detection - Live'),
        save_detections=save_detections,
        save_path=viz_config.get('save_path', 'results/live_detections'),
        
        # Logging
        log_fps_interval=log_config.get('log_fps_interval', 30),
        log_detections=log_config.get('log_detections', True),
        log_empty_frames=log_config.get('log_empty_frames', False)
    )


def on_pothole_detected(detection_frame: DetectionFrame) -> None:
    """
    Callback when potholes are detected.
    This integrates with the event system.
    """
    logger = logging.getLogger(__name__)
    
    for detection in detection_frame.detections:
        conf = detection.get('confidence', 0)
        bbox = detection.get('bbox', [])
        
        # Calculate approximate severity based on bounding box size
        if bbox:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            frame_area = detection_frame.frame.shape[0] * detection_frame.frame.shape[1]
            area_ratio = area / frame_area if frame_area > 0 else 0
            
            if conf > 0.7 and area_ratio > 0.1:
                severity = "HIGH"
            elif conf > 0.5 or area_ratio > 0.05:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            logger.info(
                f"🚨 POTHOLE DETECTED | "
                f"Severity: {severity} | "
                f"Confidence: {conf:.2%} | "
                f"Area: {area_ratio:.2%} of frame | "
                f"Frame: {detection_frame.frame_number}"
            )


def main():
    """Main entry point for live detection."""
    parser = argparse.ArgumentParser(
        description='Real-time Pothole Detection using YOLOv8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_detection.py                    # Run with default settings
  python live_detection.py --camera 1         # Use camera index 1
  python live_detection.py --model best.pt    # Use custom model
  python live_detection.py --confidence 0.5   # Set confidence threshold
  python live_detection.py --no-display       # Run headless (no window)
  python live_detection.py --save             # Save detection frames
        """
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=None,
        help='Camera device index (default: from config)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to YOLO model weights (default: from config)'
    )
    parser.add_argument(
        '--confidence', '-conf',
        type=float,
        default=None,
        help='Detection confidence threshold (default: from config)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without GUI window'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save frames with detections'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    Path('logs').mkdir(exist_ok=True)
    logger = setup_logging(args.log_level)
    
    logger.info("=" * 60)
    logger.info("  REAL-TIME POTHOLE DETECTION SYSTEM")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        live_config = create_live_config(config, args)
        
        logger.info(f"Camera: Device {live_config.camera_device_id}")
        logger.info(f"Resolution: {live_config.camera_width}x{live_config.camera_height}")
        logger.info(f"Model: {live_config.model_path}")
        logger.info(f"Confidence Threshold: {live_config.confidence_threshold}")
        logger.info(f"Display Window: {live_config.show_window}")
        logger.info(f"Save Detections: {live_config.save_detections}")
        
        # Initialize YOLOv8 detector
        logger.info("Initializing YOLO detector...")
        detector = YOLOv8Detector(
            model_path=live_config.model_path,
            confidence_threshold=live_config.confidence_threshold,
            iou_threshold=live_config.iou_threshold,
            device='cuda' if live_config.use_gpu else 'cpu'
        )
        
        # Create live detection service
        service = LiveDetectionService(
            detector=detector,
            config=live_config,
            on_detection=on_pothole_detected
        )
        
        # Run detection loop
        logger.info("")
        logger.info("Starting live detection...")
        logger.info("Press 'q' to quit, 's' to save current frame")
        logger.info("")
        
        service.run()
        
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()
