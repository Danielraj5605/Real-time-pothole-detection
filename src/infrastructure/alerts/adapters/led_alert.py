"""
LED Alert Channel
Controls LED via GPIO for Raspberry Pi
"""
import logging
from ..interfaces.alert_channel_interface import AlertChannelInterface
from ....domain.entities.alert import Alert, AlertLevel


class LEDAlertChannel(AlertChannelInterface):
    """
    LED alert channel for Raspberry Pi.
    Requires RPi.GPIO library.
    """
    
    def __init__(self, gpio_pin: int = 27):
        """
        Initialize LED channel.
        
        Args:
            gpio_pin: GPIO pin number for LED
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
            self.logger.info(f"LED initialized on GPIO pin {gpio_pin}")
        except ImportError:
            self.logger.warning("RPi.GPIO not available - LED disabled")
        except Exception as e:
            self.logger.error(f"Failed to initialize LED: {e}")
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert via LED"""
        if not self._gpio_available:
            return False
        
        try:
            # Different blink patterns for different severity levels
            if alert.level == AlertLevel.CRITICAL:
                self._blink_pattern([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Fast blinks
            elif alert.level == AlertLevel.WARNING:
                self._blink_pattern([0.3, 0.3, 0.3])  # Medium blinks
            else:
                self._blink_pattern([1.0])  # Single long blink
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send LED alert: {e}")
            return False
    
    def _blink_pattern(self, durations: list):
        """Execute blink pattern"""
        import time
        
        for duration in durations:
            self._gpio.output(self.gpio_pin, self._gpio.HIGH)
            time.sleep(duration)
            self._gpio.output(self.gpio_pin, self._gpio.LOW)
            time.sleep(0.1)
    
    def is_available(self) -> bool:
        """Check if LED is available"""
        return self._gpio_available
    
    def cleanup(self):
        """Cleanup GPIO"""
        if self._gpio_available and self._gpio:
            self._gpio.cleanup(self.gpio_pin)
