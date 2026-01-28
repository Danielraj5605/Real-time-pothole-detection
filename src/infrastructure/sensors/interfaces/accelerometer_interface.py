"""
Accelerometer Interface
"""
from abc import abstractmethod
from dataclasses import dataclass
from .sensor_interface import SensorInterface


@dataclass
class AccelerationReading:
    """3-axis acceleration data"""
    x: float  # g-force
    y: float  # g-force
    z: float  # g-force
    magnitude: float = 0.0
    
    def __post_init__(self):
        """Calculate magnitude"""
        self.magnitude = (self.x**2 + self.y**2 + self.z**2) ** 0.5


class AccelerometerInterface(SensorInterface[AccelerationReading]):
    """Abstract interface for accelerometer sensors"""
    
    @abstractmethod
    def set_range(self, range_g: int) -> bool:
        """
        Set accelerometer measurement range.
        
        Args:
            range_g: Range in g-force (e.g., 2, 4, 8, 16)
            
        Returns:
            True if range was set successfully
        """
        pass
    
    @abstractmethod
    def set_sample_rate(self, rate_hz: int) -> bool:
        """
        Set sampling rate.
        
        Args:
            rate_hz: Sample rate in Hz
            
        Returns:
            True if rate was set successfully
        """
        pass
