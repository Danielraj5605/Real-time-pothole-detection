"""
NEO-6M GPS Adapter
Wraps GPS sensor to implement GPSInterface
"""
from datetime import datetime
from typing import Optional
import logging
import serial

from ..interfaces.gps_interface import GPSInterface, GPSReading
from ..interfaces.sensor_interface import SensorReading


class NEO6MGPS(GPSInterface):
    """
    NEO-6M GPS adapter.
    Implements GPSInterface for NEO-6M GPS module.
    
    Note: Requires pyserial library
    """
    
    def __init__(self, port: str = '/dev/ttyAMA0', baudrate: int = 9600):
        """
        Initialize GPS.
        
        Args:
            port: Serial port (e.g., '/dev/ttyAMA0' for Raspberry Pi)
            baudrate: Baud rate (default 9600 for NEO-6M)
        """
        self.port = port
        self.baudrate = baudrate
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
        self._serial: Optional[serial.Serial] = None
        self._last_reading: Optional[GPSReading] = None
    
    def initialize(self) -> bool:
        """Initialize GPS module"""
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0
            )
            self._is_initialized = True
            self.logger.info(f"GPS initialized on {self.port}")
            return True
        except ImportError:
            self.logger.error("pyserial not installed. Install with: pip install pyserial")
            return False
        except Exception as e:
            self.logger.error(f"GPS initialization error: {e}")
            return False
    
    def calibrate(self) -> bool:
        """Calibrate GPS (wait for initial fix)"""
        return self.wait_for_fix(timeout_seconds=60)
    
    def read(self) -> SensorReading[GPSReading]:
        """Read current GPS position"""
        reading = self._read_gps_data()
        quality = 1.0 if (reading and reading.is_valid()) else 0.0
        
        if reading:
            self._last_reading = reading
        
        return SensorReading(
            timestamp=datetime.utcnow(),
            data=reading or self._last_reading or GPSReading(0, 0),
            quality=quality
        )
    
    def _read_gps_data(self) -> Optional[GPSReading]:
        """Read and parse GPS data from serial"""
        if not self._is_initialized or self._serial is None:
            return None
        
        try:
            # Read NMEA sentences
            for _ in range(10):  # Try up to 10 lines
                line = self._serial.readline().decode('ascii', errors='ignore').strip()
                
                # Parse GPGGA sentence (Global Positioning System Fix Data)
                if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                    return self._parse_gpgga(line)
            
            return None
        except Exception as e:
            self.logger.error(f"Error reading GPS data: {e}")
            return None
    
    def _parse_gpgga(self, sentence: str) -> Optional[GPSReading]:
        """Parse GPGGA NMEA sentence"""
        try:
            parts = sentence.split(',')
            
            if len(parts) < 15:
                return None
            
            # Check fix quality
            fix_quality = int(parts[6]) if parts[6] else 0
            if fix_quality == 0:
                return None
            
            # Parse latitude
            lat_str = parts[2]
            lat_dir = parts[3]
            if lat_str and lat_dir:
                lat_deg = float(lat_str[:2])
                lat_min = float(lat_str[2:])
                latitude = lat_deg + (lat_min / 60.0)
                if lat_dir == 'S':
                    latitude = -latitude
            else:
                return None
            
            # Parse longitude
            lon_str = parts[4]
            lon_dir = parts[5]
            if lon_str and lon_dir:
                lon_deg = float(lon_str[:3])
                lon_min = float(lon_str[3:])
                longitude = lon_deg + (lon_min / 60.0)
                if lon_dir == 'W':
                    longitude = -longitude
            else:
                return None
            
            # Parse altitude
            altitude = float(parts[9]) if parts[9] else None
            
            # Parse satellites
            satellites = int(parts[7]) if parts[7] else 0
            
            # Parse HDOP
            hdop = float(parts[8]) if parts[8] else None
            
            return GPSReading(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                satellites=satellites,
                hdop=hdop
            )
        except Exception as e:
            self.logger.error(f"Error parsing GPGGA: {e}")
            return None
    
    def wait_for_fix(self, timeout_seconds: int = 60) -> bool:
        """Wait for GPS fix"""
        import time
        
        if not self._is_initialized:
            return False
        
        self.logger.info(f"Waiting for GPS fix (timeout: {timeout_seconds}s)...")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout_seconds:
            reading = self._read_gps_data()
            if reading and reading.is_valid():
                self.logger.info(f"GPS fix acquired: {reading.satellites} satellites")
                return True
            time.sleep(1)
        
        self.logger.warning("GPS fix timeout")
        return False
    
    def has_fix(self) -> bool:
        """Check if GPS has valid fix"""
        reading = self._read_gps_data()
        return reading is not None and reading.is_valid()
    
    def is_healthy(self) -> bool:
        """Check if GPS is functioning"""
        if not self._is_initialized or self._serial is None:
            return False
        return self._serial.is_open
    
    def cleanup(self) -> None:
        """Release GPS resources"""
        if self._serial is not None:
            self._serial.close()
            self._is_initialized = False
            self.logger.info("GPS released")
