"""
Base Event Class
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Event:
    """Base event class for all domain events"""
    timestamp: datetime
    source: str
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
