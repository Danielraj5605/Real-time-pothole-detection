"""
OpenCV Camera Adapter - Enhanced
Wraps OpenCV VideoCapture with comprehensive configuration and error handling.
Supports webcams and Raspberry Pi cameras via OpenCV backend.
"""
import cv2
import time
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
import logging

from ..interfaces.camera_interface import CameraInterface, Frame
from ..interfaces.sensor_interface import SensorReading


class OpenCVCamera(CameraInterface):
    """
    Enhanced OpenCV-based camera adapter.
    
    Features:
    - Configurable camera device and settings
    - Warm-up and calibration support
    - FPS monitoring
    - Automatic reconnection on failure
    - Backend selection (V4L2, DSHOW, etc.)
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: Optional[int] = None,
        warmup_frames: int = 10,
        buffer_size: int = 1,
        auto_reconnect: bool = True
    ):
        """
        Initialize camera.
        
        Args:
            camera_id: Camera device ID (0 for default camera)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            backend: OpenCV backend (e.g., cv2.CAP_DSHOW for Windows)
            warmup_frames: Number of frames to capture during warmup
            buffer_size: Camera buffer size (lower = less latency)
            auto_reconnect: Attempt to reconnect on failure
        """
        self.camera_id = camera_id
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.backend = backend
        self.warmup_frames = warmup_frames
        self.buffer_size = buffer_size
        self.auto_reconnect = auto_reconnect
        
        self.capture: Optional[cv2.VideoCapture] = None
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
        
        # Actual camera properties (after initialization)
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0
        
        # Performance tracking
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._failed_reads = 0
        self._max_failed_reads = 10
    
    def initialize(self) -> bool:
        """
        Initialize camera hardware.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info(f"Initializing camera {self.camera_id}...")
            
            # Open camera with optional backend
            if self.backend is not None:
                self.capture = cv2.VideoCapture(self.camera_id, self.backend)
            else:
                self.capture = cv2.VideoCapture(self.camera_id)
            
            if not self.capture.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Configure camera properties
            self._configure_camera()
            
            # Verify properties
            self._read_actual_properties()
            
            self._is_initialized = True
            self._start_time = time.time()
            self._failed_reads = 0
            
            self.logger.info(
                f"✓ Camera initialized: {self.actual_width}x{self.actual_height} @ {self.actual_fps} FPS"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False
    
    def _configure_camera(self) -> None:
        """Configure camera properties."""
        if self.capture is None:
            return
        
        # Set resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        
        # Set FPS
        self.capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Minimize buffer for low latency
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        
        # Additional settings for better performance
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    def _read_actual_properties(self) -> None:
        """Read actual camera properties after configuration."""
        if self.capture is None:
            return
        
        self.actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        
        if self.actual_width != self.target_width or self.actual_height != self.target_height:
            self.logger.warning(
                f"Requested {self.target_width}x{self.target_height}, "
                f"got {self.actual_width}x{self.actual_height}"
            )
    
    def calibrate(self) -> bool:
        """
        Calibrate camera by capturing warmup frames.
        
        Returns:
            True if calibration successful
        """
        if not self._is_initialized:
            self.logger.error("Cannot calibrate: camera not initialized")
            return False
        
        self.logger.info(f"Warming up camera ({self.warmup_frames} frames)...")
        
        for i in range(self.warmup_frames):
            ret, _ = self.capture.read()
            if not ret:
                self.logger.warning(f"Failed to read warmup frame {i+1}")
        
        self.logger.info("✓ Camera calibrated")
        return True
    
    def read(self) -> SensorReading[Frame]:
        """
        Read current frame as SensorReading.
        
        Returns:
            SensorReading containing Frame data
        """
        frame = self.capture_frame()
        quality = 1.0 if frame is not None else 0.0
        
        return SensorReading(
            timestamp=datetime.utcnow(),
            data=frame,
            quality=quality
        )
    
    def capture_frame(self) -> Optional[Frame]:
        """
        Capture a single frame.
        
        Returns:
            Frame object or None if capture failed
        """
        if not self._is_initialized or self.capture is None:
            return None
        
        ret, image = self.capture.read()
        
        if not ret or image is None:
            self._failed_reads += 1
            
            if self._failed_reads >= self._max_failed_reads:
                self.logger.warning(
                    f"Too many failed reads ({self._failed_reads}). "
                    f"Attempting reconnect..."
                )
                if self.auto_reconnect:
                    self._reconnect()
            
            return None
        
        # Reset failed read counter on success
        self._failed_reads = 0
        self._frame_count += 1
        
        height, width = image.shape[:2]
        return Frame(
            image=image,
            width=width,
            height=height,
            format="BGR"
        )
    
    def capture_frame_with_timestamp(self) -> Tuple[Optional[Frame], datetime]:
        """
        Capture frame with precise timestamp.
        
        Returns:
            Tuple of (Frame, timestamp)
        """
        timestamp = datetime.utcnow()
        frame = self.capture_frame()
        return frame, timestamp
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to camera."""
        self.logger.info("Attempting to reconnect to camera...")
        
        # Release current capture
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        
        self._is_initialized = False
        time.sleep(0.5)  # Brief delay before reconnect
        
        # Reinitialize
        success = self.initialize()
        if success:
            self.logger.info("✓ Camera reconnected successfully")
            self._failed_reads = 0
        else:
            self.logger.error("✗ Camera reconnection failed")
        
        return success
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            
        Returns:
            True if resolution was set successfully
        """
        if not self._is_initialized:
            return False
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self._read_actual_properties()
        
        success = (self.actual_width == width and self.actual_height == height)
        if success:
            self.logger.info(f"Resolution set to {width}x{height}")
        else:
            self.logger.warning(
                f"Resolution change failed: wanted {width}x{height}, "
                f"got {self.actual_width}x{self.actual_height}"
            )
        
        return success
    
    def set_fps(self, fps: int) -> bool:
        """
        Set camera frame rate.
        
        Args:
            fps: Frames per second
            
        Returns:
            True if FPS was set successfully
        """
        if not self._is_initialized:
            return False
        
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        
        success = (self.actual_fps == fps)
        if success:
            self.logger.info(f"FPS set to {fps}")
        else:
            self.logger.warning(f"FPS change: wanted {fps}, got {self.actual_fps}")
        
        return success
    
    def set_exposure(self, exposure: float) -> bool:
        """
        Set camera exposure.
        
        Args:
            exposure: Exposure value (camera-dependent)
            
        Returns:
            True if exposure was set
        """
        if not self._is_initialized:
            return False
        
        # Disable auto exposure first
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
        self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
        
        return True
    
    def set_auto_exposure(self, enabled: bool) -> bool:
        """
        Enable or disable auto exposure.
        
        Args:
            enabled: True to enable auto exposure
            
        Returns:
            True if setting was applied
        """
        if not self._is_initialized:
            return False
        
        # Values are camera/driver dependent
        # 0.25 = manual, 0.75 = auto (common on Linux)
        value = 0.75 if enabled else 0.25
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
        
        return True
    
    def is_healthy(self) -> bool:
        """
        Check if camera is functioning.
        
        Returns:
            True if camera is working properly
        """
        if not self._is_initialized or self.capture is None:
            return False
        
        if not self.capture.isOpened():
            return False
        
        if self._failed_reads >= self._max_failed_reads:
            return False
        
        return True
    
    def get_fps(self) -> float:
        """
        Get actual measured FPS.
        
        Returns:
            Frames per second
        """
        if self._start_time is None or self._frame_count == 0:
            return 0.0
        
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        
        return self._frame_count / elapsed
    
    def get_properties(self) -> dict:
        """
        Get current camera properties.
        
        Returns:
            Dictionary of camera properties
        """
        return {
            'camera_id': self.camera_id,
            'width': self.actual_width,
            'height': self.actual_height,
            'fps': self.actual_fps,
            'measured_fps': self.get_fps(),
            'frame_count': self._frame_count,
            'failed_reads': self._failed_reads,
            'is_initialized': self._is_initialized,
            'is_healthy': self.is_healthy()
        }
    
    def cleanup(self) -> None:
        """Release camera resources."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self._is_initialized = False
            self.logger.info("Camera released")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        self.calibrate()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False


class RaspberryPiCamera(OpenCVCamera):
    """
    Raspberry Pi camera adapter.
    Uses OpenCV with GStreamer or PiCamera2 backend.
    """
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        use_picamera2: bool = False,
        sensor_mode: int = 0
    ):
        """
        Initialize Raspberry Pi camera.
        
        Args:
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            use_picamera2: Use PiCamera2 library (requires installation)
            sensor_mode: Camera sensor mode
        """
        self.use_picamera2 = use_picamera2
        self.sensor_mode = sensor_mode
        
        # GStreamer pipeline for Pi camera
        # This works with libcamera on Raspberry Pi OS Bullseye+
        self.gstreamer_pipeline = None
        
        super().__init__(
            camera_id=0,
            width=width,
            height=height,
            fps=fps,
            backend=cv2.CAP_V4L2,  # Use V4L2 backend on Linux
            warmup_frames=10
        )
    
    def initialize(self) -> bool:
        """Initialize Raspberry Pi camera."""
        self.logger.info("Initializing Raspberry Pi camera...")
        
        if self.use_picamera2:
            return self._initialize_picamera2()
        else:
            return self._initialize_gstreamer()
    
    def _initialize_gstreamer(self) -> bool:
        """Initialize camera using GStreamer pipeline."""
        try:
            # GStreamer pipeline for libcamera
            pipeline = (
                f"libcamerasrc ! "
                f"video/x-raw,width={self.target_width},"
                f"height={self.target_height},"
                f"framerate={self.target_fps}/1 ! "
                f"videoconvert ! video/x-raw,format=BGR ! "
                f"appsink drop=1"
            )
            
            self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.capture.isOpened():
                # Fallback to standard camera
                self.logger.warning("GStreamer pipeline failed, falling back to V4L2")
                return super().initialize()
            
            self._read_actual_properties()
            self._is_initialized = True
            self._start_time = time.time()
            
            self.logger.info(
                f"✓ Pi camera initialized via GStreamer: "
                f"{self.actual_width}x{self.actual_height}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"GStreamer initialization failed: {e}")
            return super().initialize()
    
    def _initialize_picamera2(self) -> bool:
        """Initialize camera using PiCamera2 library."""
        try:
            from picamera2 import Picamera2
            
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (self.target_width, self.target_height)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            
            self._is_initialized = True
            self._start_time = time.time()
            
            self.logger.info("✓ Pi camera initialized via PiCamera2")
            return True
            
        except ImportError:
            self.logger.error("PiCamera2 not installed. Install with: pip install picamera2")
            return False
        except Exception as e:
            self.logger.error(f"PiCamera2 initialization failed: {e}")
            return False
    
    def capture_frame(self) -> Optional[Frame]:
        """Capture frame from Pi camera."""
        if self.use_picamera2 and hasattr(self, 'picam2'):
            try:
                image = self.picam2.capture_array()
                # PiCamera2 returns RGB, convert to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                self._frame_count += 1
                height, width = image.shape[:2]
                
                return Frame(
                    image=image,
                    width=width,
                    height=height,
                    format="BGR"
                )
            except Exception as e:
                self.logger.error(f"PiCamera2 capture error: {e}")
                return None
        else:
            return super().capture_frame()
    
    def cleanup(self) -> None:
        """Release Pi camera resources."""
        if self.use_picamera2 and hasattr(self, 'picam2'):
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                self.logger.error(f"PiCamera2 cleanup error: {e}")
        
        super().cleanup()
