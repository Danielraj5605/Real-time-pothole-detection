"""
GPS Interface
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
from .sensor_interface import SensorInterface


@dataclass
class GPSReading:
    """GPS position data"""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    speed: Optional[float] = None
    heading: Optional[float] = None
    satellites: int = 0
    hdop: Optional[float] = None  # Horizontal Dilution of Precision
    
    def is_valid(self) -> bool:
        """Check if GPS reading is valid"""
        return (
            -90 <= self.latitude <= 90 and
            -180 <= self.longitude <= 180 and
            self.satellites >= 3
        )


class GPSInterface(SensorInterface[GPSReading]):
    """Abstract interface for GPS sensors"""
    
    @abstractmethod
    def wait_for_fix(self, timeout_seconds: int = 60) -> bool:
        """
        Wait for GPS fix.
        
        Args:
            timeout_seconds: Maximum time to wait
            
        Returns:
            True if fix acquired
        """
        pass
    
    @abstractmethod
    def has_fix(self) -> bool:
        """
        Check if GPS has a valid fix.
        
        Returns:
            True if GPS has fix
        """
        pass
