"""
NEO-6M GPS Driver for Raspberry Pi Zero WH
Lightweight UART driver for GPS data collection.
Parses NMEA sentences for location, speed, and heading.

Hardware: NEO-6M connected via UART (TXD: GPIO14, RXD: GPIO15)
Baud Rate: 9600
"""

import time
import threading

try:
    import serial
except ImportError:
    print("Error: pyserial not installed. Run: pip install pyserial")
    raise

try:
    import pynmea2
except ImportError:
    print("Error: pynmea2 not installed. Run: pip install pynmea2")
    raise


class GPS:
    """
    Lightweight NEO-6M GPS driver.
    Reads and parses NMEA sentences from serial port.
    """
    
    def __init__(self, port='/dev/serial0', baudrate=9600, timeout=1.0):
        """
        Initialize GPS.
        
        Args:
            port: Serial port (default /dev/serial0 for Pi Zero WH)
            baudrate: Baud rate (NEO-6M default is 9600)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        
        # Current GPS data (thread-safe access via lock)
        self._lock = threading.Lock()
        self._latitude = None
        self._longitude = None
        self._altitude = None
        self._speed_kmh = None
        self._heading = None
        self._hdop = None
        self._fix_quality = 0
        self._satellites = 0
        self._last_update = 0
        
        # Background reader thread
        self._running = False
        self._reader_thread = None
    
    def connect(self):
        """
        Open the serial connection to the GPS.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"GPS connected on {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            print(f"Failed to connect to GPS: {e}")
            return False
    
    def start_reading(self):
        """Start the background reader thread."""
        if self._running:
            return
        
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        print("GPS reader thread started")
    
    def stop_reading(self):
        """Stop the background reader thread."""
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
    
    def _reader_loop(self):
        """Background thread that continuously reads and parses GPS data."""
        while self._running:
            try:
                if self.serial and self.serial.in_waiting:
                    line = self.serial.readline()
                    if line:
                        try:
                            # Decode and parse NMEA sentence
                            sentence = line.decode('ascii', errors='ignore').strip()
                            if sentence.startswith('$'):
                                self._parse_nmea(sentence)
                        except Exception:
                            pass  # Ignore parsing errors
                else:
                    time.sleep(0.01)  # Prevent busy loop
            except Exception:
                time.sleep(0.1)
    
    def _parse_nmea(self, sentence):
        """Parse an NMEA sentence and update internal state."""
        try:
            msg = pynmea2.parse(sentence)
            
            # GGA - Global Positioning System Fix Data
            if isinstance(msg, pynmea2.GGA):
                with self._lock:
                    if msg.latitude and msg.longitude:
                        self._latitude = msg.latitude
                        self._longitude = msg.longitude
                        self._altitude = msg.altitude if msg.altitude else 0.0
                        self._fix_quality = int(msg.gps_qual) if msg.gps_qual else 0
                        self._satellites = int(msg.num_sats) if msg.num_sats else 0
                        self._hdop = float(msg.horizontal_dil) if msg.horizontal_dil else 99.9
                        self._last_update = time.time()
            
            # RMC - Recommended Minimum Navigation Information
            elif isinstance(msg, pynmea2.RMC):
                with self._lock:
                    if msg.latitude and msg.longitude:
                        self._latitude = msg.latitude
                        self._longitude = msg.longitude
                        self._last_update = time.time()
                    
                    # Speed in knots, convert to km/h
                    if msg.spd_over_grnd:
                        self._speed_kmh = float(msg.spd_over_grnd) * 1.852
                    
                    # True heading
                    if msg.true_course:
                        self._heading = float(msg.true_course)
            
            # VTG - Track Made Good and Ground Speed
            elif isinstance(msg, pynmea2.VTG):
                with self._lock:
                    if hasattr(msg, 'spd_over_grnd_kmph') and msg.spd_over_grnd_kmph:
                        self._speed_kmh = float(msg.spd_over_grnd_kmph)
                    if hasattr(msg, 'true_track') and msg.true_track:
                        self._heading = float(msg.true_track)
        
        except pynmea2.ParseError:
            pass  # Ignore malformed sentences
        except Exception:
            pass
    
    def read_once(self):
        """
        Read and parse a single NMEA sentence (blocking).
        Useful for testing without the background thread.
        
        Returns:
            Parsed message or None
        """
        if not self.serial:
            return None
        
        try:
            line = self.serial.readline()
            if line:
                sentence = line.decode('ascii', errors='ignore').strip()
                if sentence.startswith('$'):
                    return pynmea2.parse(sentence)
        except Exception:
            pass
        return None
    
    def get_position(self):
        """
        Get current GPS position.
        
        Returns:
            Dict with latitude, longitude, altitude, or None if no fix
        """
        with self._lock:
            if self._latitude is None or self._longitude is None:
                return None
            
            return {
                'latitude': self._latitude,
                'longitude': self._longitude,
                'altitude_m': self._altitude or 0.0,
                'fix_quality': self._fix_quality,
                'satellites': self._satellites,
                'hdop': self._hdop or 99.9
            }
    
    def get_motion(self):
        """
        Get current speed and heading.
        
        Returns:
            Dict with speed_kmh and heading_deg, or None if unavailable
        """
        with self._lock:
            return {
                'speed_kmh': self._speed_kmh or 0.0,
                'heading_deg': self._heading or 0.0
            }
    
    def get_all(self):
        """
        Get all GPS data.
        
        Returns:
            Dict with all GPS fields, or None values if no fix
        """
        with self._lock:
            return {
                'latitude': self._latitude,
                'longitude': self._longitude,
                'altitude_m': self._altitude or 0.0,
                'speed_kmh': self._speed_kmh or 0.0,
                'heading_deg': self._heading or 0.0,
                'hdop': self._hdop or 99.9,
                'fix_quality': self._fix_quality,
                'satellites': self._satellites,
                'last_update': self._last_update
            }
    
    def has_fix(self):
        """
        Check if GPS has a valid fix.
        
        Returns:
            True if GPS has a fix, False otherwise
        """
        with self._lock:
            return self._fix_quality > 0 and self._latitude is not None
    
    def wait_for_fix(self, timeout=120, check_interval=1.0):
        """
        Wait for GPS to acquire a fix.
        
        Args:
            timeout: Maximum time to wait in seconds
            check_interval: How often to check for fix
            
        Returns:
            True if fix acquired, False if timeout
        """
        print(f"Waiting for GPS fix (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.has_fix():
                pos = self.get_position()
                print(f"\nGPS fix acquired!")
                print(f"  Position: {pos['latitude']:.6f}, {pos['longitude']:.6f}")
                print(f"  Satellites: {pos['satellites']}, HDOP: {pos['hdop']:.1f}")
                return True
            
            # Show progress
            elapsed = int(time.time() - start_time)
            with self._lock:
                sats = self._satellites
            print(f"  Searching... ({elapsed}s, {sats} satellites)", end='\r')
            
            time.sleep(check_interval)
        
        print(f"\nGPS fix timeout after {timeout}s")
        return False
    
    def close(self):
        """Close the GPS connection."""
        self.stop_reading()
        if self.serial:
            self.serial.close()
            self.serial = None
        print("GPS connection closed")


# Test function
if __name__ == "__main__":
    print("NEO-6M GPS Driver Test")
    print("=" * 40)
    
    gps = GPS()
    
    if not gps.connect():
        print("Failed to connect to GPS!")
        exit(1)
    
    # Start background reading
    gps.start_reading()
    
    try:
        # Wait for fix
        if gps.wait_for_fix(timeout=300):  # 5 minutes for cold start
            print("\nReading GPS data (Ctrl+C to stop)...\n")
            
            while True:
                data = gps.get_all()
                
                if data['latitude']:
                    print(f"Lat: {data['latitude']:+10.6f}  "
                          f"Lon: {data['longitude']:+11.6f}  "
                          f"Alt: {data['altitude_m']:6.1f}m  "
                          f"Speed: {data['speed_kmh']:5.1f} km/h  "
                          f"Heading: {data['heading_deg']:5.1f}°  "
                          f"HDOP: {data['hdop']:.1f}  "
                          f"Sats: {data['satellites']}", end='\r')
                else:
                    print("Waiting for fix...", end='\r')
                
                time.sleep(1.0)
        else:
            print("Could not get GPS fix. Make sure you have a clear view of the sky.")
    
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        gps.close()
