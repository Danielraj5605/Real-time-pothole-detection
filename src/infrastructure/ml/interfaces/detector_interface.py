"""
Detector Interface - Abstract ML Detection
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Any
import numpy as np


@dataclass
class DetectionResult:
    """Single detection result"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def area(self) -> int:
        """Calculate bounding box area"""
        x1, y1, x2, y2 = self.bbox
        return int((x2 - x1) * (y2 - y1))
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'area': self.area
        }


class DetectorInterface(ABC):
    """Abstract interface for ML detectors"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector model"""
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run detection on an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection results
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release model resources"""
        pass
