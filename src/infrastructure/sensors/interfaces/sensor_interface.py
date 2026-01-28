"""
Base Sensor Interface - DRY Principle
All sensors implement this common interface
"""
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass
from datetime import datetime

T = TypeVar('T')


@dataclass
class SensorReading(Generic[T]):
    """Generic sensor reading wrapper"""
    timestamp: datetime
    data: T
    quality: float  # 0.0 - 1.0, where 1.0 is perfect


class SensorInterface(ABC, Generic[T]):
    """
    Base interface for all sensors - DRY principle.
    Ensures consistent behavior across all sensor types.
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize sensor hardware.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """
        Calibrate sensor for accurate readings.
        
        Returns:
            True if calibration successful
        """
        pass
    
    @abstractmethod
    def read(self) -> SensorReading[T]:
        """
        Read current sensor data.
        
        Returns:
            SensorReading with timestamp, data, and quality
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if sensor is functioning correctly.
        
        Returns:
            True if sensor is healthy
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release sensor resources"""
        pass
