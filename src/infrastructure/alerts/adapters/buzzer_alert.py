"""
Buzzer Alert Channel
Controls buzzer via GPIO for Raspberry Pi
"""
import logging
from ..interfaces.alert_channel_interface import AlertChannelInterface
from ....domain.entities.alert import Alert, AlertLevel


class BuzzerAlertChannel(AlertChannelInterface):
    """
    Buzzer alert channel for Raspberry Pi.
    Requires RPi.GPIO library.
    """
    
    def __init__(self, gpio_pin: int = 17):
        """
        Initialize buzzer channel.
        
        Args:
            gpio_pin: GPIO pin number for buzzer
        """
        self.gpio_pin = gpio_pin
        self.logger = logging.getLogger(__name__)
        self._gpio_available = False
        self._gpio = None
        
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            self._gpio.setmode(GPIO.BCM)
            self._gpio.setup(self.gpio_pin, GPIO.OUT)
            self._gpio_available = True
            self.logger.info(f"Buzzer initialized on GPIO pin {gpio_pin}")
        except ImportError:
            self.logger.warning("RPi.GPIO not available - buzzer disabled")
        except Exception as e:
            self.logger.error(f"Failed to initialize buzzer: {e}")
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via buzzer"""
        if not self._gpio_available:
            return False
        
        try:
            # Different beep patterns for different severity levels
            if alert.level == AlertLevel.CRITICAL:
                self._beep_pattern([0.1, 0.1, 0.1, 0.1, 0.1])  # Rapid beeps
            elif alert.level == AlertLevel.WARNING:
                self._beep_pattern([0.2, 0.2, 0.2])  # Medium beeps
            else:
                self._beep_pattern([0.5])  # Single beep
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send buzzer alert: {e}")
            return False
    
    def _beep_pattern(self, durations: list):
        """Execute beep pattern"""
        import time
        
        for duration in durations:
            self._gpio.output(self.gpio_pin, self._gpio.HIGH)
            time.sleep(duration)
            self._gpio.output(self.gpio_pin, self._gpio.LOW)
            time.sleep(0.1)
    
    def is_available(self) -> bool:
        """Check if buzzer is available"""
        return self._gpio_available
    
    def cleanup(self):
        """Cleanup GPIO"""
        if self._gpio_available and self._gpio:
            self._gpio.cleanup(self.gpio_pin)
