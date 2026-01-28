"""
Live Detection Service - Real-Time Camera Pothole Detection

Handles the complete real-time detection pipeline:
- Camera initialization and frame capture
- YOLO inference on live frames
- FPS monitoring and performance metrics
- Visualization with bounding boxes
- Integration with fusion and event bus
"""
import cv2
import time
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Protocol
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DetectionFrame:
    """Container for a detection result frame"""
    frame: np.ndarray
    detections: List[Dict[str, Any]]
    timestamp: datetime
    fps: float
    frame_number: int


@dataclass
class LiveDetectionConfig:
    """Configuration for live detection"""
    # Camera settings
    camera_device_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    warmup_frames: int = 10
    
    # Model settings
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Processing settings
    target_fps: int = 15
    skip_frames: int = 0
    resize_width: int = 640
    resize_height: int = 480
    use_gpu: bool = True
    
    # Visualization
    show_window: bool = True
    show_fps: bool = True
    show_detections: bool = True
    show_confidence: bool = True
    bbox_color: tuple = (0, 255, 0)
    bbox_thickness: int = 2
    font_scale: float = 0.6
    window_name: str = "Pothole Detection - Live"
    save_detections: bool = False
    save_path: str = "results/live_detections"
    
    # Logging
    log_fps_interval: int = 30
    log_detections: bool = True
    log_empty_frames: bool = False


class DetectorPort(Protocol):
    """Protocol for ML detector"""
    def detect(self, image: np.ndarray) -> List[Dict]: ...
    def initialize(self) -> bool: ...
    def is_loaded(self) -> bool: ...


class LiveDetectionService:
    """
    Real-time camera-based pothole detection service.
    
    Features:
    - Live camera capture with OpenCV
    - YOLOv8 inference on each frame
    - FPS tracking and performance monitoring
    - Bounding box visualization
    - Event integration for detected potholes
    - Graceful error handling and shutdown
    """
    
    def __init__(
        self,
        detector: DetectorPort,
        config: LiveDetectionConfig,
        on_detection: Optional[Callable[[DetectionFrame], None]] = None
    ):
        """
        Initialize live detection service.
        
        Args:
            detector: ML detector implementation (YOLOv8)
            config: Live detection configuration
            on_detection: Callback when potholes are detected
        """
        self.detector = detector
        self.config = config
        self.on_detection = on_detection
        self.logger = logging.getLogger(__name__)
        
        # Camera state
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_initialized = False
        
        # Performance metrics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time: Optional[float] = None
        self.fps_history: List[float] = []
        self.last_fps_log_time = 0
        
        # Frame timings for FPS calculation
        self._frame_times: List[float] = []
        self._max_frame_history = 30
    
    def initialize(self) -> bool:
        """
        Initialize camera and detector.
        
        Returns:
            True if initialization successful
        """
        self.logger.info("Initializing live detection service...")
        
        # Initialize detector first
        if not self.detector.is_loaded():
            self.logger.info("Initializing YOLO detector...")
            if not self.detector.initialize():
                self.logger.error("Failed to initialize detector")
                return False
            self.logger.info("✓ Detector initialized")
        
        # Initialize camera
        if not self._initialize_camera():
            self.logger.error("Failed to initialize camera")
            return False
        
        self.is_initialized = True
        self.logger.info("✓ Live detection service initialized successfully")
        return True
    
    def _initialize_camera(self) -> bool:
        """Initialize and configure the camera."""
        try:
            self.logger.info(f"Opening camera device {self.config.camera_device_id}...")
            
            # Open camera
            self.camera = cv2.VideoCapture(self.config.camera_device_id)
            
            if not self.camera.isOpened():
                self.logger.error(f"Failed to open camera {self.config.camera_device_id}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(
                f"Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS"
            )
            
            # Warm up camera
            self.logger.info(f"Warming up camera ({self.config.warmup_frames} frames)...")
            for _ in range(self.config.warmup_frames):
                ret, _ = self.camera.read()
                if not ret:
                    self.logger.warning("Failed to read warmup frame")
            
            self.logger.info("✓ Camera initialized and warmed up")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            BGR image as numpy array, or None if failed
        """
        if self.camera is None or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None
        
        # Resize if needed
        if (frame.shape[1] != self.config.resize_width or 
            frame.shape[0] != self.config.resize_height):
            frame = cv2.resize(
                frame, 
                (self.config.resize_width, self.config.resize_height)
            )
        
        return frame
    
    def detect_on_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLO detection on a single frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        if not self.detector.is_loaded():
            return []
        
        try:
            detections = self.detector.detect(frame)
            return [
                {
                    'bbox': d.bbox if hasattr(d, 'bbox') else d.get('bbox'),
                    'confidence': d.confidence if hasattr(d, 'confidence') else d.get('confidence'),
                    'class_name': d.class_name if hasattr(d, 'class_name') else d.get('class_name', 'pothole'),
                    'class_id': d.class_id if hasattr(d, 'class_id') else d.get('class_id', 0)
                }
                for d in detections
            ]
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def visualize_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        fps: float
    ) -> np.ndarray:
        """
        Draw bounding boxes and info on frame.
        
        Args:
            frame: Input frame (will be modified in place)
            detections: List of detection dictionaries
            fps: Current FPS for display
            
        Returns:
            Annotated frame
        """
        display_frame = frame.copy()
        
        # Draw detections
        if self.config.show_detections:
            for det in detections:
                bbox = det.get('bbox')
                if bbox is None:
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                conf = det.get('confidence', 0)
                class_name = det.get('class_name', 'pothole')
                
                # Draw bounding box
                cv2.rectangle(
                    display_frame,
                    (x1, y1), (x2, y2),
                    self.config.bbox_color,
                    self.config.bbox_thickness
                )
                
                # Draw label
                if self.config.show_confidence:
                    label = f"{class_name}: {conf:.2f}"
                else:
                    label = class_name
                
                # Label background
                (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale, 1
                )
                cv2.rectangle(
                    display_frame,
                    (x1, y1 - h - 10), (x1 + w, y1),
                    self.config.bbox_color, -1
                )
                cv2.putText(
                    display_frame, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    (255, 255, 255), 1
                )
        
        # Draw FPS
        if self.config.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                display_frame, fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
        
        # Draw detection count
        det_text = f"Detections: {len(detections)}"
        cv2.putText(
            display_frame, det_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
        
        # Draw total potholes detected
        total_text = f"Total Potholes: {self.detection_count}"
        cv2.putText(
            display_frame, total_text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )
        
        return display_frame
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS from frame timing history."""
        current_time = time.time()
        self._frame_times.append(current_time)
        
        # Keep only recent frames
        if len(self._frame_times) > self._max_frame_history:
            self._frame_times = self._frame_times[-self._max_frame_history:]
        
        if len(self._frame_times) < 2:
            return 0.0
        
        time_diff = self._frame_times[-1] - self._frame_times[0]
        if time_diff <= 0:
            return 0.0
        
        return (len(self._frame_times) - 1) / time_diff
    
    def run(self) -> None:
        """
        Run the live detection loop.
        
        Press 'q' to quit, 's' to save current frame.
        """
        if not self.is_initialized:
            if not self.initialize():
                self.logger.error("Failed to initialize. Cannot run.")
                return
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        skip_counter = 0
        
        self.logger.info("=" * 50)
        self.logger.info("LIVE POTHOLE DETECTION STARTED")
        self.logger.info("=" * 50)
        self.logger.info("Press 'q' to quit, 's' to save frame")
        self.logger.info("")
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    if self.config.log_empty_frames:
                        self.logger.warning("Empty frame captured")
                    continue
                
                self.frame_count += 1
                
                # Skip frames if configured
                skip_counter += 1
                if skip_counter <= self.config.skip_frames:
                    continue
                skip_counter = 0
                
                # Run detection
                detections = self.detect_on_frame(frame)
                
                # Update detection count
                if detections:
                    self.detection_count += len(detections)
                    if self.config.log_detections:
                        self.logger.info(
                            f"[Frame {self.frame_count}] Detected {len(detections)} pothole(s)"
                        )
                
                # Calculate FPS
                fps = self._calculate_fps()
                
                # Log FPS periodically
                current_time = time.time()
                if current_time - self.last_fps_log_time >= self.config.log_fps_interval:
                    self.logger.info(f"Performance: {fps:.1f} FPS, {self.frame_count} frames processed")
                    self.last_fps_log_time = current_time
                
                # Create detection frame
                detection_frame = DetectionFrame(
                    frame=frame,
                    detections=detections,
                    timestamp=datetime.now(),
                    fps=fps,
                    frame_number=self.frame_count
                )
                
                # Call detection callback
                if detections and self.on_detection:
                    try:
                        self.on_detection(detection_frame)
                    except Exception as e:
                        self.logger.error(f"Detection callback error: {e}")
                
                # Visualize
                if self.config.show_window:
                    display_frame = self.visualize_detections(frame, detections, fps)
                    cv2.imshow(self.config.window_name, display_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.logger.info("Quit requested by user")
                        break
                    elif key == ord('s'):
                        self._save_frame(frame, detections)
                
                # Save detections if configured
                if self.config.save_detections and detections:
                    self._save_frame(frame, detections)
                
                # Frame rate limiting
                elapsed = time.time() - loop_start
                target_frame_time = 1.0 / self.config.target_fps
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.stop()
    
    async def run_async(self) -> None:
        """Async version of run loop."""
        if not self.is_initialized:
            if not self.initialize():
                self.logger.error("Failed to initialize. Cannot run.")
                return
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        self.logger.info("ASYNC LIVE DETECTION STARTED")
        
        try:
            while self.is_running:
                # Capture and process frame
                frame = self.capture_frame()
                if frame is None:
                    await asyncio.sleep(0.001)
                    continue
                
                self.frame_count += 1
                detections = self.detect_on_frame(frame)
                
                if detections:
                    self.detection_count += len(detections)
                
                fps = self._calculate_fps()
                
                # Visualization in async mode
                if self.config.show_window:
                    display_frame = self.visualize_detections(frame, detections, fps)
                    cv2.imshow(self.config.window_name, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Allow other tasks to run
                await asyncio.sleep(0.001)
                
        except asyncio.CancelledError:
            self.logger.info("Async loop cancelled")
        finally:
            self.stop()
    
    def _save_frame(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """Save frame with detections to disk."""
        try:
            save_dir = Path(self.config.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"detection_{timestamp}.jpg"
            filepath = save_dir / filename
            
            # Draw detections on frame before saving
            annotated = self.visualize_detections(
                frame, detections, self._calculate_fps()
            )
            
            cv2.imwrite(str(filepath), annotated)
            self.logger.info(f"Saved detection frame: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save frame: {e}")
    
    def stop(self) -> None:
        """Stop the detection loop and cleanup resources."""
        self.is_running = False
        
        # Print summary
        if self.start_time:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            self.logger.info("")
            self.logger.info("=" * 50)
            self.logger.info("LIVE DETECTION STOPPED")
            self.logger.info("=" * 50)
            self.logger.info(f"Total frames processed: {self.frame_count}")
            self.logger.info(f"Total potholes detected: {self.detection_count}")
            self.logger.info(f"Average FPS: {avg_fps:.1f}")
            self.logger.info(f"Total runtime: {elapsed:.1f} seconds")
            self.logger.info("=" * 50)
        
        # Cleanup
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.logger.info("Camera released")
        
        cv2.destroyAllWindows()
        self.is_initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'elapsed_time': elapsed,
            'average_fps': self.frame_count / elapsed if elapsed > 0 else 0,
            'current_fps': self._calculate_fps(),
            'is_running': self.is_running
        }
