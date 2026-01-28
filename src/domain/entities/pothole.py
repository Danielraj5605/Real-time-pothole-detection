"""
Pothole Entity - Core Domain Model
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid
from math import radians, sin, cos, sqrt, atan2


class Severity(Enum):
    """Pothole severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    
    @classmethod
    def from_metrics(cls, accel_peak: float, confidence: float, bbox_area: int) -> 'Severity':
        """
        Business logic for severity classification - Single Responsibility
        
        Args:
            accel_peak: Peak acceleration value (g-force)
            confidence: ML detection confidence (0-1)
            bbox_area: Bounding box area in pixels
            
        Returns:
            Severity level based on combined metrics
        """
        if accel_peak > 2.5 and confidence > 0.7 and bbox_area > 10000:
            return cls.HIGH
        elif accel_peak > 1.8 and confidence > 0.6 and bbox_area > 5000:
            return cls.MEDIUM
        return cls.LOW


@dataclass
class Pothole:
    """
    Core domain entity representing a detected pothole.
    Immutable after creation (frozen=True can be added if needed).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    latitude: float = 0.0
    longitude: float = 0.0
    severity: Severity = Severity.LOW
    confidence: float = 0.0
    accel_peak: float = 0.0
    bbox_area: int = 0
    image_path: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)
    is_verified: bool = False
    
    def distance_to(self, lat: float, lon: float) -> float:
        """
        Calculate distance to a point using Haversine formula.
        
        Args:
            lat: Target latitude
            lon: Target longitude
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth's radius in meters
        lat1, lon1 = radians(self.latitude), radians(self.longitude)
        lat2, lon2 = radians(lat), radians(lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def should_alert_at_distance(self, distance_m: float) -> bool:
        """
        Business rule for alerting based on severity and distance.
        
        Args:
            distance_m: Distance in meters
            
        Returns:
            True if alert should be triggered
        """
        thresholds = {
            Severity.HIGH: 100,
            Severity.MEDIUM: 50,
            Severity.LOW: 20
        }
        return distance_m <= thresholds.get(self.severity, 20)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'accel_peak': self.accel_peak,
            'bbox_area': self.bbox_area,
            'image_path': self.image_path,
            'detected_at': self.detected_at.isoformat(),
            'is_verified': self.is_verified
        }
