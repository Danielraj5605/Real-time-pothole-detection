"""
OpenCV Camera Adapter
Wraps OpenCV VideoCapture to implement CameraInterface
"""
import cv2
import numpy as np
from datetime import datetime
from typing import Optional
import logging

from ..interfaces.camera_interface import CameraInterface, Frame
from ..interfaces.sensor_interface import SensorReading


class OpenCVCamera(CameraInterface):
    """
    OpenCV-based camera adapter.
    Implements CameraInterface using cv2.VideoCapture.
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize camera.
        
        Args:
            camera_id: Camera device ID (0 for default camera)
        """
        self.camera_id = camera_id
        self.capture: Optional[cv2.VideoCapture] = None
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize camera hardware"""
        try:
            self.capture = cv2.VideoCapture(self.camera_id)
            if not self.capture.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            self._is_initialized = True
            self.logger.info(f"Camera {self.camera_id} initialized")
            return True
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False
    
    def calibrate(self) -> bool:
        """Calibrate camera (basic implementation)"""
        if not self._is_initialized:
            return False
        
        # Warm up camera by capturing a few frames
        for _ in range(5):
            self.capture.read()
        
        self.logger.info("Camera calibrated")
        return True
    
    def read(self) -> SensorReading[Frame]:
        """Read current frame"""
        frame = self.capture_frame()
        quality = 1.0 if frame is not None else 0.0
        
        return SensorReading(
            timestamp=datetime.utcnow(),
            data=frame,
            quality=quality
        )
    
    def capture_frame(self) -> Optional[Frame]:
        """Capture a single frame"""
        if not self._is_initialized or self.capture is None:
            return None
        
        ret, image = self.capture.read()
        if not ret or image is None:
            return None
        
        height, width = image.shape[:2]
        return Frame(
            image=image,
            width=width,
            height=height,
            format="BGR"
        )
    
    def set_resolution(self, width: int, height: int) -> bool:
        """Set camera resolution"""
        if not self._is_initialized:
            return False
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verify
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        success = (actual_width == width and actual_height == height)
        if success:
            self.logger.info(f"Resolution set to {width}x{height}")
        else:
            self.logger.warning(
                f"Requested {width}x{height}, got {actual_width}x{actual_height}"
            )
        
        return success
    
    def set_fps(self, fps: int) -> bool:
        """Set camera frame rate"""
        if not self._is_initialized:
            return False
        
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        
        success = (actual_fps == fps)
        if success:
            self.logger.info(f"FPS set to {fps}")
        else:
            self.logger.warning(f"Requested {fps} FPS, got {actual_fps}")
        
        return success
    
    def is_healthy(self) -> bool:
        """Check if camera is functioning"""
        if not self._is_initialized or self.capture is None:
            return False
        return self.capture.isOpened()
    
    def cleanup(self) -> None:
        """Release camera resources"""
        if self.capture is not None:
            self.capture.release()
            self._is_initialized = False
            self.logger.info("Camera released")
