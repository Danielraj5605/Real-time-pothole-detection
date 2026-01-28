"""
Mock Detector for Testing
"""
import logging
from typing import List
import numpy as np

from ..interfaces.detector_interface import DetectorInterface, DetectionResult


class MockDetector(DetectorInterface):
    """Mock detector for testing without ML model"""
    
    def __init__(self, detection_probability: float = 0.3):
        """
        Initialize mock detector.
        
        Args:
            detection_probability: Probability of detecting a pothole (0-1)
        """
        self.logger = logging.getLogger(__name__)
        self.detection_probability = detection_probability
        self._is_loaded = False
    
    def initialize(self) -> bool:
        """Initialize mock detector"""
        self._is_loaded = True
        self.logger.info("Mock detector initialized")
        return True
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Simulate detection.
        
        Args:
            image: Input image (not actually used)
            
        Returns:
            Random detection results
        """
        if not self._is_loaded:
            return []
        
        # Randomly decide if we detect a pothole
        if np.random.random() < self.detection_probability:
            # Generate random detection
            height, width = image.shape[:2]
            
            # Random bbox
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = x1 + np.random.randint(50, width // 2)
            y2 = y1 + np.random.randint(50, height // 2)
            
            # Random confidence
            confidence = np.random.uniform(0.5, 0.95)
            
            detection = DetectionResult(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                confidence=confidence,
                class_id=0,
                class_name="pothole"
            )
            
            return [detection]
        
        return []
    
    def is_loaded(self) -> bool:
        """Check if detector is loaded"""
        return self._is_loaded
    
    def cleanup(self) -> None:
        """Release resources"""
        self._is_loaded = False
        self.logger.info("Mock detector released")
