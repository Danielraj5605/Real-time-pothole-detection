"""
MPU6050 Accelerometer Adapter
Wraps MPU6050 sensor to implement AccelerometerInterface
"""
from datetime import datetime
from typing import Optional
import logging

from ..interfaces.accelerometer_interface import AccelerometerInterface, AccelerationReading
from ..interfaces.sensor_interface import SensorReading


class MPU6050Accelerometer(AccelerometerInterface):
    """
    MPU6050 accelerometer adapter.
    Implements AccelerometerInterface for MPU6050 sensor.
    
    Note: Requires smbus2 library and I2C enabled on Raspberry Pi
    """
    
    def __init__(self, i2c_address: int = 0x68, i2c_bus: int = 1):
        """
        Initialize MPU6050.
        
        Args:
            i2c_address: I2C address of MPU6050 (default 0x68)
            i2c_bus: I2C bus number (default 1 for Raspberry Pi)
        """
        self.i2c_address = i2c_address
        self.i2c_bus = i2c_bus
        self.logger = logging.getLogger(__name__)
        self._is_initialized = False
        self._bus = None
        self._accel_scale = 16384.0  # For ±2g range
    
    def initialize(self) -> bool:
        """Initialize MPU6050 sensor"""
        try:
            from smbus2 import SMBus
            
            self._bus = SMBus(self.i2c_bus)
            
            # Wake up MPU6050 (it starts in sleep mode)
            self._bus.write_byte_data(self.i2c_address, 0x6B, 0)
            
            self._is_initialized = True
            self.logger.info(f"MPU6050 initialized at address 0x{self.i2c_address:02x}")
            return True
            
        except ImportError:
            self.logger.error("smbus2 not installed. Install with: pip install smbus2")
            return False
        except Exception as e:
            self.logger.error(f"MPU6050 initialization error: {e}")
            return False
    
    def calibrate(self) -> bool:
        """Calibrate accelerometer"""
        if not self._is_initialized:
            return False
        
        # Simple calibration: read several samples to stabilize
        try:
            for _ in range(10):
                self._read_raw_data()
            
            self.logger.info("MPU6050 calibrated")
            return True
        except Exception as e:
            self.logger.error(f"Calibration error: {e}")
            return False
    
    def read(self) -> SensorReading[AccelerationReading]:
        """Read current acceleration"""
        try:
            accel_data = self._read_raw_data()
            quality = 1.0 if accel_data is not None else 0.0
            
            return SensorReading(
                timestamp=datetime.utcnow(),
                data=accel_data,
                quality=quality
            )
        except Exception as e:
            self.logger.error(f"Read error: {e}")
            return SensorReading(
                timestamp=datetime.utcnow(),
                data=AccelerationReading(0, 0, 0),
                quality=0.0
            )
    
    def _read_raw_data(self) -> Optional[AccelerationReading]:
        """Read raw acceleration data from sensor"""
        if not self._is_initialized or self._bus is None:
            return None
        
        try:
            # Read accelerometer data (registers 0x3B to 0x40)
            data = self._bus.read_i2c_block_data(self.i2c_address, 0x3B, 6)
            
            # Convert to signed 16-bit values
            accel_x_raw = self._bytes_to_int16(data[0], data[1])
            accel_y_raw = self._bytes_to_int16(data[2], data[3])
            accel_z_raw = self._bytes_to_int16(data[4], data[5])
            
            # Convert to g-force
            accel_x = accel_x_raw / self._accel_scale
            accel_y = accel_y_raw / self._accel_scale
            accel_z = accel_z_raw / self._accel_scale
            
            return AccelerationReading(
                x=accel_x,
                y=accel_y,
                z=accel_z
            )
        except Exception as e:
            self.logger.error(f"Error reading data: {e}")
            return None
    
    @staticmethod
    def _bytes_to_int16(high_byte: int, low_byte: int) -> int:
        """Convert two bytes to signed 16-bit integer"""
        value = (high_byte << 8) | low_byte
        if value >= 0x8000:
            value = -((65535 - value) + 1)
        return value
    
    def set_range(self, range_g: int) -> bool:
        """Set accelerometer measurement range"""
        if not self._is_initialized:
            return False
        
        # MPU6050 range settings
        range_map = {
            2: (0x00, 16384.0),
            4: (0x08, 8192.0),
            8: (0x10, 4096.0),
            16: (0x18, 2048.0)
        }
        
        if range_g not in range_map:
            self.logger.error(f"Invalid range: {range_g}. Use 2, 4, 8, or 16")
            return False
        
        try:
            reg_value, scale = range_map[range_g]
            self._bus.write_byte_data(self.i2c_address, 0x1C, reg_value)
            self._accel_scale = scale
            self.logger.info(f"Accelerometer range set to ±{range_g}g")
            return True
        except Exception as e:
            self.logger.error(f"Error setting range: {e}")
            return False
    
    def set_sample_rate(self, rate_hz: int) -> bool:
        """Set sampling rate"""
        if not self._is_initialized:
            return False
        
        # MPU6050 sample rate = Gyroscope Output Rate / (1 + SMPLRT_DIV)
        # Gyroscope Output Rate = 8kHz when DLPF is disabled (DLPF_CFG = 0 or 7)
        # Gyroscope Output Rate = 1kHz when DLPF is enabled
        
        try:
            # Assuming 1kHz base rate (DLPF enabled)
            sample_rate_div = max(0, min(255, (1000 // rate_hz) - 1))
            self._bus.write_byte_data(self.i2c_address, 0x19, sample_rate_div)
            
            actual_rate = 1000 // (1 + sample_rate_div)
            self.logger.info(f"Sample rate set to {actual_rate}Hz")
            return True
        except Exception as e:
            self.logger.error(f"Error setting sample rate: {e}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if sensor is functioning"""
        if not self._is_initialized:
            return False
        
        try:
            # Try to read WHO_AM_I register
            who_am_i = self._bus.read_byte_data(self.i2c_address, 0x75)
            return who_am_i == 0x68
        except:
            return False
    
    def cleanup(self) -> None:
        """Release sensor resources"""
        if self._bus is not None:
            self._bus.close()
            self._is_initialized = False
            self.logger.info("MPU6050 released")
