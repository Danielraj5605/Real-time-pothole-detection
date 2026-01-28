"""
Pothole Detected Event
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np

from .base_event import Event
from ...domain.entities.pothole import Pothole


@dataclass
class PotholeDetectedEvent(Event):
    """Event fired when a pothole is detected"""
    pothole: Pothole
    raw_frame: Optional[np.ndarray] = None
    
    def __init__(self, pothole: Pothole, raw_frame: Optional[np.ndarray] = None):
        super().__init__(
            timestamp=datetime.utcnow(),
            source="detection_service"
        )
        self.pothole = pothole
        self.raw_frame = raw_frame
