"""
Alert Triggered Event
"""
from dataclasses import dataclass
from datetime import datetime

from .base_event import Event
from ...domain.entities.alert import Alert


@dataclass
class AlertTriggeredEvent(Event):
    """Event fired when an alert is triggered"""
    alert: Alert
    
    def __init__(self, alert: Alert):
        super().__init__(
            timestamp=datetime.utcnow(),
            source="alert_service"
        )
        self.alert = alert
