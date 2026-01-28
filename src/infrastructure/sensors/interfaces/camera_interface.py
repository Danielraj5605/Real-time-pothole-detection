"""
Camera Interface
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .sensor_interface import SensorInterface


@dataclass
class Frame:
    """Camera frame data"""
    image: np.ndarray
    width: int
    height: int
    format: str = "BGR"  # OpenCV default


class CameraInterface(SensorInterface[Frame]):
    """Abstract interface for camera sensors"""
    
    @abstractmethod
    def capture_frame(self) -> Optional[Frame]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Frame object or None if capture failed
        """
        pass
    
    @abstractmethod
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set camera resolution.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            
        Returns:
            True if resolution was set successfully
        """
        pass
    
    @abstractmethod
    def set_fps(self, fps: int) -> bool:
        """
        Set camera frame rate.
        
        Args:
            fps: Frames per second
            
        Returns:
            True if FPS was set successfully
        """
        pass
