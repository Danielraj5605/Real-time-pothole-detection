"""
Mock Sensors for Testing
Allows testing without physical hardware
"""
import numpy as np
from datetime import datetime
from typing import Optional
import logging
import time

from ..interfaces.camera_interface import CameraInterface, Frame
from ..interfaces.accelerometer_interface import AccelerometerInterface, AccelerationReading
from ..interfaces.gps_interface import GPSInterface, GPSReading
from ..interfaces.sensor_interface import SensorReading


class MockCamera(CameraInterface):
    """Mock camera for testing without hardware"""
    
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
        self._fps = 30
    
    def initialize(self) -> bool:
        self._is_initialized = True
        self.logger.info("Mock camera initialized")
        return True
    
    def calibrate(self) -> bool:
        self.logger.info("Mock camera calibrated")
        return True
    
    def read(self) -> SensorReading[Frame]:
        frame = self.capture_frame()
        return SensorReading(
            timestamp=datetime.utcnow(),
            data=frame,
            quality=1.0
        )
    
    def capture_frame(self) -> Optional[Frame]:
        """Generate a random test frame"""
        if not self._is_initialized:
            return None
        
        # Generate random noise image
        image = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        
        return Frame(
            image=image,
            width=self.width,
            height=self.height,
            format="BGR"
        )
    
    def set_resolution(self, width: int, height: int) -> bool:
        self.width = width
        self.height = height
        self.logger.info(f"Mock camera resolution set to {width}x{height}")
        return True
    
    def set_fps(self, fps: int) -> bool:
        self._fps = fps
        self.logger.info(f"Mock camera FPS set to {fps}")
        return True
    
    def is_healthy(self) -> bool:
        return self._is_initialized
    
    def cleanup(self) -> None:
        self._is_initialized = False
        self.logger.info("Mock camera released")


class MockAccelerometer(AccelerometerInterface):
    """Mock accelerometer for testing without hardware"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
        self._range_g = 2
        self._sample_rate = 50
        self._pothole_mode = False
    
    def initialize(self) -> bool:
        self._is_initialized = True
        self.logger.info("Mock accelerometer initialized")
        return True
    
    def calibrate(self) -> bool:
        self.logger.info("Mock accelerometer calibrated")
        return True
    
    def read(self) -> SensorReading[AccelerationReading]:
        """Generate simulated acceleration data"""
        if not self._is_initialized:
            accel = AccelerationReading(0, 0, 0)
        else:
            # Normal: small random variations around gravity
            # Pothole mode: occasional spikes
            if self._pothole_mode and np.random.random() < 0.1:
                # Simulate pothole impact
                x = np.random.uniform(-3.0, 3.0)
                y = np.random.uniform(-3.0, 3.0)
                z = np.random.uniform(8.0, 12.0)  # Large spike
            else:
                # Normal road vibrations
                x = np.random.uniform(-0.5, 0.5)
                y = np.random.uniform(-0.5, 0.5)
                z = np.random.uniform(9.3, 10.3)  # Around 1g (9.8 m/s²)
            
            accel = AccelerationReading(x, y, z)
        
        return SensorReading(
            timestamp=datetime.utcnow(),
            data=accel,
            quality=1.0
        )
    
    def set_range(self, range_g: int) -> bool:
        self._range_g = range_g
        self.logger.info(f"Mock accelerometer range set to ±{range_g}g")
        return True
    
    def set_sample_rate(self, rate_hz: int) -> bool:
        self._sample_rate = rate_hz
        self.logger.info(f"Mock accelerometer sample rate set to {rate_hz}Hz")
        return True
    
    def enable_pothole_mode(self, enabled: bool = True):
        """Enable pothole simulation mode"""
        self._pothole_mode = enabled
        self.logger.info(f"Pothole mode: {'enabled' if enabled else 'disabled'}")
    
    def is_healthy(self) -> bool:
        return self._is_initialized
    
    def cleanup(self) -> None:
        self._is_initialized = False
        self.logger.info("Mock accelerometer released")


class MockGPS(GPSInterface):
    """Mock GPS for testing without hardware"""
    
    def __init__(self, start_lat: float = 37.7749, start_lon: float = -122.4194):
        """
        Initialize mock GPS.
        
        Args:
            start_lat: Starting latitude (default: San Francisco)
            start_lon: Starting longitude
        """
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
        self._has_fix = False
        self._latitude = start_lat
        self._longitude = start_lon
        self._speed = 0.0
    
    def initialize(self) -> bool:
        self._is_initialized = True
        self.logger.info("Mock GPS initialized")
        return True
    
    def calibrate(self) -> bool:
        self._has_fix = True
        self.logger.info("Mock GPS calibrated (fix acquired)")
        return True
    
    def read(self) -> SensorReading[GPSReading]:
        """Generate simulated GPS data"""
        if not self._is_initialized or not self._has_fix:
            reading = GPSReading(0, 0, satellites=0)
            quality = 0.0
        else:
            # Simulate movement (small random walk)
            self._latitude += np.random.uniform(-0.0001, 0.0001)
            self._longitude += np.random.uniform(-0.0001, 0.0001)
            self._speed = np.random.uniform(0, 60)  # 0-60 km/h
            
            reading = GPSReading(
                latitude=self._latitude,
                longitude=self._longitude,
                altitude=np.random.uniform(0, 100),
                speed=self._speed,
                heading=np.random.uniform(0, 360),
                satellites=np.random.randint(4, 12),
                hdop=np.random.uniform(0.5, 2.0)
            )
            quality = 1.0
        
        return SensorReading(
            timestamp=datetime.utcnow(),
            data=reading,
            quality=quality
        )
    
    def wait_for_fix(self, timeout_seconds: int = 60) -> bool:
        """Simulate waiting for GPS fix"""
        self.logger.info("Simulating GPS fix acquisition...")
        time.sleep(0.5)  # Simulate short delay
        self._has_fix = True
        self.logger.info("Mock GPS fix acquired")
        return True
    
    def has_fix(self) -> bool:
        return self._has_fix
    
    def is_healthy(self) -> bool:
        return self._is_initialized
    
    def cleanup(self) -> None:
        self._is_initialized = False
        self._has_fix = False
        self.logger.info("Mock GPS released")
