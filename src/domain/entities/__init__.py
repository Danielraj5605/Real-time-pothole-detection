"""Domain Entities"""

from .pothole import Pothole, Severity
from .sensor_data import SensorData, AccelerometerData, GPSData
from .alert import Alert, AlertLevel

__all__ = [
    'Pothole',
    'Severity',
    'SensorData',
    'AccelerometerData',
    'GPSData',
    'Alert',
    'AlertLevel'
]
