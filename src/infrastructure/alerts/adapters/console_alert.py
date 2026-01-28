"""
Console Alert Channel
Prints alerts to console with color coding
"""
import logging
from ..interfaces.alert_channel_interface import AlertChannelInterface
from ....domain.entities.alert import Alert, AlertLevel


class ConsoleAlertChannel(AlertChannelInterface):
    """
    Console alert channel.
    Prints alerts to console with ANSI color codes.
    """
    
    # ANSI color codes
    COLORS = {
        AlertLevel.INFO: '\033[94m',      # Blue
        AlertLevel.WARNING: '\033[93m',    # Yellow
        AlertLevel.CRITICAL: '\033[91m'    # Red
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, use_colors: bool = True):
        """
        Initialize console alert channel.
        
        Args:
            use_colors: Whether to use ANSI color codes
        """
        self.use_colors = use_colors
        self.logger = logging.getLogger(__name__)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert to console"""
        try:
            if self.use_colors:
                color = self.COLORS.get(alert.level, '')
                print(f"\n{self.BOLD}{color}🚨 ALERT: {alert.level.value}{self.RESET}")
                print(f"{color}   Message: {alert.message}{self.RESET}")
                print(f"{color}   Distance: {alert.distance_meters:.1f}m{self.RESET}\n")
            else:
                print(f"\n🚨 ALERT: {alert.level.value}")
                print(f"   Message: {alert.message}")
                print(f"   Distance: {alert.distance_meters:.1f}m\n")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send console alert: {e}")
            return False
    
    def is_available(self) -> bool:
        """Console is always available"""
        return True
