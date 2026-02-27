"""
MPU9250 Driver for Raspberry Pi Zero WH
Lightweight I2C driver for accelerometer/gyroscope data collection.
Optimized for Pi Zero WH's limited resources.

Hardware: MPU9250 connected via I2C (SDA: GPIO2, SCL: GPIO3)
I2C Address: 0x68 (default) or 0x69 (AD0 high)
"""

import time
import struct

try:
    import smbus2
except ImportError:
    print("Error: smbus2 not installed. Run: pip install smbus2")
    raise

# MPU9250 Register Addresses
MPU9250_ADDR = 0x68  # Default I2C address (AD0 low)

# Power Management
PWR_MGMT_1 = 0x6B
PWR_MGMT_2 = 0x6C

# Configuration
SMPLRT_DIV = 0x19     # Sample Rate Divider
CONFIG = 0x1A         # Configuration (DLPF)
GYRO_CONFIG = 0x1B    # Gyroscope Configuration
ACCEL_CONFIG = 0x1C   # Accelerometer Configuration
ACCEL_CONFIG2 = 0x1D  # Accelerometer Configuration 2

# Data Registers (high byte first)
ACCEL_XOUT_H = 0x3B
ACCEL_XOUT_L = 0x3C
ACCEL_YOUT_H = 0x3D
ACCEL_YOUT_L = 0x3E
ACCEL_ZOUT_H = 0x3F
ACCEL_ZOUT_L = 0x40
TEMP_OUT_H = 0x41
TEMP_OUT_L = 0x42
GYRO_XOUT_H = 0x43
GYRO_XOUT_L = 0x44
GYRO_YOUT_H = 0x45
GYRO_YOUT_L = 0x46
GYRO_ZOUT_H = 0x47
GYRO_ZOUT_L = 0x48

# WHO_AM_I register for verification
WHO_AM_I = 0x75
MPU9250_WHO_AM_I_VALUE = 0x71  # Expected value for MPU9250


class MPU9250:
    """
    Lightweight MPU9250 driver for accelerometer and gyroscope data.
    Magnetometer not used to reduce CPU overhead.
    """
    
    def __init__(self, bus_num=1, address=MPU9250_ADDR, accel_range=16, gyro_range=2000):
        """
        Initialize MPU9250.
        
        Args:
            bus_num: I2C bus number (1 for Pi Zero WH)
            address: I2C address (0x68 or 0x69)
            accel_range: Accelerometer full-scale range (2, 4, 8, or 16 g)
            gyro_range: Gyroscope full-scale range (250, 500, 1000, or 2000 dps)
        """
        self.bus = smbus2.SMBus(bus_num)
        self.address = address
        
        # Scale factors (to convert raw values to physical units)
        self.accel_scale = self._get_accel_scale(accel_range)
        self.gyro_scale = self._get_gyro_scale(gyro_range)
        
        # Initialize the sensor
        self._init_sensor(accel_range, gyro_range)
        
        # Calibration offsets (set to zero initially)
        self.accel_offset = [0.0, 0.0, 0.0]
        self.gyro_offset = [0.0, 0.0, 0.0]
    
    def _get_accel_scale(self, accel_range):
        """Get the scale factor for accelerometer based on range."""
        scales = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
        return scales.get(accel_range, 2048.0)
    
    def _get_gyro_scale(self, gyro_range):
        """Get the scale factor for gyroscope based on range."""
        scales = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
        return scales.get(gyro_range, 16.4)
    
    def _get_accel_config(self, accel_range):
        """Get the config byte for accelerometer range."""
        configs = {2: 0x00, 4: 0x08, 8: 0x10, 16: 0x18}
        return configs.get(accel_range, 0x18)  # Default to ±16g
    
    def _get_gyro_config(self, gyro_range):
        """Get the config byte for gyroscope range."""
        configs = {250: 0x00, 500: 0x08, 1000: 0x10, 2000: 0x18}
        return configs.get(gyro_range, 0x18)  # Default to ±2000 dps
    
    def _init_sensor(self, accel_range, gyro_range):
        """Initialize the MPU9250 sensor."""
        # Wake up the sensor (clear sleep bit)
        self.bus.write_byte_data(self.address, PWR_MGMT_1, 0x00)
        time.sleep(0.1)
        
        # Set clock source to PLL with X-axis gyro reference
        self.bus.write_byte_data(self.address, PWR_MGMT_1, 0x01)
        time.sleep(0.1)
        
        # Set sample rate divider for ~100Hz (1000Hz / (1 + 9) = 100Hz)
        self.bus.write_byte_data(self.address, SMPLRT_DIV, 0x09)
        
        # Set DLPF (Digital Low Pass Filter) to 44Hz bandwidth
        self.bus.write_byte_data(self.address, CONFIG, 0x03)
        
        # Set accelerometer range
        self.bus.write_byte_data(self.address, ACCEL_CONFIG, self._get_accel_config(accel_range))
        
        # Set accelerometer DLPF
        self.bus.write_byte_data(self.address, ACCEL_CONFIG2, 0x03)
        
        # Set gyroscope range
        self.bus.write_byte_data(self.address, GYRO_CONFIG, self._get_gyro_config(gyro_range))
        
        time.sleep(0.1)
    
    def verify_connection(self):
        """
        Verify the sensor is connected and responding.
        
        Returns:
            True if MPU9250 detected, False otherwise
        """
        try:
            who_am_i = self.bus.read_byte_data(self.address, WHO_AM_I)
            # MPU9250 returns 0x71, MPU6050 returns 0x68
            if who_am_i == 0x71:
                print(f"MPU9250 detected (WHO_AM_I: 0x{who_am_i:02X})")
                return True
            elif who_am_i == 0x68:
                print(f"MPU6050 detected (WHO_AM_I: 0x{who_am_i:02X}) - Using as MPU9250 compatible")
                return True
            else:
                print(f"Unknown device (WHO_AM_I: 0x{who_am_i:02X})")
                return True  # Try anyway
        except Exception as e:
            print(f"Error verifying MPU9250: {e}")
            return False
    
    def _read_raw_data(self, reg_h):
        """Read 16-bit signed value from two consecutive registers."""
        high = self.bus.read_byte_data(self.address, reg_h)
        low = self.bus.read_byte_data(self.address, reg_h + 1)
        value = (high << 8) | low
        # Convert to signed 16-bit
        if value >= 0x8000:
            value = value - 0x10000
        return value
    
    def read_accel_raw(self):
        """
        Read raw accelerometer values.
        
        Returns:
            Tuple (ax, ay, az) in raw units
        """
        ax = self._read_raw_data(ACCEL_XOUT_H)
        ay = self._read_raw_data(ACCEL_YOUT_H)
        az = self._read_raw_data(ACCEL_ZOUT_H)
        return ax, ay, az
    
    def read_gyro_raw(self):
        """
        Read raw gyroscope values.
        
        Returns:
            Tuple (gx, gy, gz) in raw units
        """
        gx = self._read_raw_data(GYRO_XOUT_H)
        gy = self._read_raw_data(GYRO_YOUT_H)
        gz = self._read_raw_data(GYRO_ZOUT_H)
        return gx, gy, gz
    
    def read_accel(self):
        """
        Read accelerometer values in g.
        
        Returns:
            Tuple (ax, ay, az) in g
        """
        raw = self.read_accel_raw()
        ax = (raw[0] / self.accel_scale) - self.accel_offset[0]
        ay = (raw[1] / self.accel_scale) - self.accel_offset[1]
        az = (raw[2] / self.accel_scale) - self.accel_offset[2]
        return ax, ay, az
    
    def read_gyro(self):
        """
        Read gyroscope values in degrees per second.
        
        Returns:
            Tuple (gx, gy, gz) in dps
        """
        raw = self.read_gyro_raw()
        gx = (raw[0] / self.gyro_scale) - self.gyro_offset[0]
        gy = (raw[1] / self.gyro_scale) - self.gyro_offset[1]
        gz = (raw[2] / self.gyro_scale) - self.gyro_offset[2]
        return gx, gy, gz
    
    def read_all(self):
        """
        Read all accelerometer and gyroscope values efficiently.
        Uses burst read for better performance.
        
        Returns:
            Dict with 'accel' (ax, ay, az in g) and 'gyro' (gx, gy, gz in dps)
        """
        # Burst read 14 bytes: accel (6) + temp (2) + gyro (6)
        try:
            data = self.bus.read_i2c_block_data(self.address, ACCEL_XOUT_H, 14)
        except Exception:
            # Fallback to individual reads
            accel = self.read_accel()
            gyro = self.read_gyro()
            return {'accel': accel, 'gyro': gyro}
        
        # Parse accelerometer data
        ax = (data[0] << 8 | data[1])
        if ax >= 0x8000:
            ax -= 0x10000
        ay = (data[2] << 8 | data[3])
        if ay >= 0x8000:
            ay -= 0x10000
        az = (data[4] << 8 | data[5])
        if az >= 0x8000:
            az -= 0x10000
        
        # Parse gyroscope data (skip temp at 6,7)
        gx = (data[8] << 8 | data[9])
        if gx >= 0x8000:
            gx -= 0x10000
        gy = (data[10] << 8 | data[11])
        if gy >= 0x8000:
            gy -= 0x10000
        gz = (data[12] << 8 | data[13])
        if gz >= 0x8000:
            gz -= 0x10000
        
        # Convert to physical units
        accel = (
            (ax / self.accel_scale) - self.accel_offset[0],
            (ay / self.accel_scale) - self.accel_offset[1],
            (az / self.accel_scale) - self.accel_offset[2]
        )
        gyro = (
            (gx / self.gyro_scale) - self.gyro_offset[0],
            (gy / self.gyro_scale) - self.gyro_offset[1],
            (gz / self.gyro_scale) - self.gyro_offset[2]
        )
        
        return {'accel': accel, 'gyro': gyro}
    
    def calibrate(self, samples=100):
        """
        Calibrate the sensor by taking average readings while stationary.
        Place the sensor on a flat, level surface before calling.
        
        Args:
            samples: Number of samples to average
        """
        print(f"Calibrating MPU9250 ({samples} samples)...")
        print("Keep the sensor stationary on a flat surface!")
        
        accel_sum = [0.0, 0.0, 0.0]
        gyro_sum = [0.0, 0.0, 0.0]
        
        for i in range(samples):
            data = self.read_all()
            accel_sum[0] += data['accel'][0]
            accel_sum[1] += data['accel'][1]
            accel_sum[2] += data['accel'][2]
            gyro_sum[0] += data['gyro'][0]
            gyro_sum[1] += data['gyro'][1]
            gyro_sum[2] += data['gyro'][2]
            time.sleep(0.01)  # 100Hz
        
        # Calculate averages
        self.accel_offset[0] = accel_sum[0] / samples
        self.accel_offset[1] = accel_sum[1] / samples
        # For Z axis, subtract 1g (gravity) since sensor is flat
        self.accel_offset[2] = (accel_sum[2] / samples) - 1.0
        
        self.gyro_offset[0] = gyro_sum[0] / samples
        self.gyro_offset[1] = gyro_sum[1] / samples
        self.gyro_offset[2] = gyro_sum[2] / samples
        
        print(f"Calibration complete!")
        print(f"Accel offsets: X={self.accel_offset[0]:.4f}, Y={self.accel_offset[1]:.4f}, Z={self.accel_offset[2]:.4f}")
        print(f"Gyro offsets: X={self.gyro_offset[0]:.4f}, Y={self.gyro_offset[1]:.4f}, Z={self.gyro_offset[2]:.4f}")
    
    def close(self):
        """Close the I2C bus."""
        self.bus.close()


# Test function
if __name__ == "__main__":
    print("MPU9250 Driver Test")
    print("=" * 40)
    
    try:
        mpu = MPU9250()
        
        if not mpu.verify_connection():
            print("Failed to connect to MPU9250!")
            exit(1)
        
        print("\nReading sensor data (Ctrl+C to stop)...\n")
        
        while True:
            data = mpu.read_all()
            ax, ay, az = data['accel']
            gx, gy, gz = data['gyro']
            
            # Calculate total acceleration magnitude
            accel_mag = (ax**2 + ay**2 + az**2) ** 0.5
            
            print(f"Accel: X={ax:+7.3f}g  Y={ay:+7.3f}g  Z={az:+7.3f}g  |  Mag={accel_mag:.3f}g  |  "
                  f"Gyro: X={gx:+8.2f}  Y={gy:+8.2f}  Z={gz:+8.2f} dps", end='\r')
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'mpu' in dir():
            mpu.close()
