"""
Alert Entity
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Alert entity for driver notifications"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pothole_id: str = ""
    level: AlertLevel = AlertLevel.INFO
    message: str = ""
    distance_meters: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'pothole_id': self.pothole_id,
            'level': self.level.value,
            'message': self.message,
            'distance_meters': self.distance_meters,
            'created_at': self.created_at.isoformat(),
            'acknowledged': self.acknowledged
        }
