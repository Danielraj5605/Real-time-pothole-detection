"""
Alert Channel Interface
"""
from abc import ABC, abstractmethod
from ....domain.entities.alert import Alert


class AlertChannelInterface(ABC):
    """Abstract interface for alert delivery channels"""
    
    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through this channel.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if channel is available.
        
        Returns:
            True if channel can send alerts
        """
        pass
