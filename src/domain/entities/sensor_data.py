"""
Sensor Data Entities
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class AccelerometerData:
    """Accelerometer sensor reading"""
    x: float
    y: float
    z: float
    timestamp: datetime
    magnitude: Optional[float] = None
    
    def __post_init__(self):
        """Calculate magnitude if not provided"""
        if self.magnitude is None:
            self.magnitude = (self.x**2 + self.y**2 + self.z**2) ** 0.5


@dataclass
class GPSData:
    """GPS sensor reading"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    speed: Optional[float] = None
    timestamp: Optional[datetime] = None
    accuracy: Optional[float] = None
    
    def is_valid(self) -> bool:
        """Check if GPS data is valid"""
        return (
            -90 <= self.latitude <= 90 and
            -180 <= self.longitude <= 180
        )


@dataclass
class SensorData:
    """Combined sensor data from all sources"""
    accelerometer: Optional[AccelerometerData] = None
    gps: Optional[GPSData] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
