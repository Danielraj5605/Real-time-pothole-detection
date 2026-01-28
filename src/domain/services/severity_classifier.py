"""
Severity Classification Service
"""
from ..entities.pothole import Severity


class SeverityClassifier:
    """
    Service for classifying pothole severity.
    Encapsulates business rules for severity determination.
    """
    
    def __init__(
        self,
        high_accel_threshold: float = 2.5,
        high_confidence_threshold: float = 0.7,
        high_bbox_threshold: int = 10000,
        medium_accel_threshold: float = 1.8,
        medium_confidence_threshold: float = 0.6,
        medium_bbox_threshold: int = 5000
    ):
        """Initialize with configurable thresholds"""
        self.high_accel = high_accel_threshold
        self.high_conf = high_confidence_threshold
        self.high_bbox = high_bbox_threshold
        self.medium_accel = medium_accel_threshold
        self.medium_conf = medium_confidence_threshold
        self.medium_bbox = medium_bbox_threshold
    
    def classify(
        self,
        accel_peak: float,
        confidence: float,
        bbox_area: int
    ) -> Severity:
        """
        Classify severity based on multiple metrics.
        
        Args:
            accel_peak: Peak acceleration (g-force)
            confidence: ML detection confidence (0-1)
            bbox_area: Bounding box area in pixels
            
        Returns:
            Severity level
        """
        if (accel_peak > self.high_accel and 
            confidence > self.high_conf and 
            bbox_area > self.high_bbox):
            return Severity.HIGH
        elif (accel_peak > self.medium_accel and 
              confidence > self.medium_conf and 
              bbox_area > self.medium_bbox):
            return Severity.MEDIUM
        return Severity.LOW
