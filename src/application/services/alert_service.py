"""
Alert Service - Manages Alert Generation and Delivery
Enhanced with history, callbacks, and statistics
"""
import logging
from typing import List, Protocol, Callable, Dict, Any, Optional
from collections import deque
from datetime import datetime

from ...domain.entities.pothole import Pothole
from ...domain.entities.alert import Alert, AlertLevel
from ...domain.services.proximity_calculator import ProximityCalculator
from ..events.event_bus import EventBus
from ..events.alert_triggered import AlertTriggeredEvent


class AlertChannelPort(Protocol):
    """Port for alert delivery channels"""
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert"""
        ...


class AlertService:
    """
    Service for managing proximity alerts.
    
    Features:
    - Proximity-based alert generation
    - Alert history management
    - Callback system for custom handlers
    - Statistics tracking
    - Alert acknowledgment
    """
    
    def __init__(
        self,
        proximity_calculator: ProximityCalculator,
        alert_channels: List[AlertChannelPort],
        event_bus: EventBus,
        max_distance_m: float = 200,
        max_history: int = 100
    ):
        """
        Initialize alert service.
        
        Args:
            proximity_calculator: Service for calculating proximity
            alert_channels: List of alert delivery channels
            event_bus: Event bus for publishing events
            max_distance_m: Maximum distance to check for potholes
            max_history: Maximum alerts to keep in history
        """
        self._proximity = proximity_calculator
        self._channels = alert_channels
        self._event_bus = event_bus
        self._max_distance = max_distance_m
        self._logger = logging.getLogger(__name__)
        
        # Alert history and callbacks
        self._history: deque = deque(maxlen=max_history)
        self._callbacks: List[Callable[[Alert], None]] = []
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """
        Register a callback for new alerts.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self._callbacks.append(callback)
        self._logger.debug(f"Added alert callback: {callback.__name__}")
    
    def remove_callback(self, callback: Callable[[Alert], None]) -> bool:
        """
        Remove a callback.
        
        Args:
            callback: Callback to remove
            
        Returns:
            True if removed, False if not found
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    async def check_proximity(
        self,
        current_lat: float,
        current_lon: float,
        known_potholes: List[Pothole]
    ) -> List[Alert]:
        """
        Check for nearby potholes and generate alerts.
        
        Args:
            current_lat: Current latitude
            current_lon: Current longitude
            known_potholes: List of known potholes
            
        Returns:
            List of generated alerts
        """
        alerts = []
        
        # Find nearby potholes
        nearby = self._proximity.find_nearby_potholes(
            current_lat,
            current_lon,
            known_potholes,
            self._max_distance
        )
        
        for pothole, distance in nearby:
            # Check if alert should be triggered
            if self._proximity.should_trigger_alert(pothole, distance):
                alert = self._create_alert(pothole, distance)
                
                # Add to history
                self._history.append(alert)
                
                # Send through all channels
                for channel in self._channels:
                    try:
                        channel.send_alert(alert)
                    except Exception as e:
                        self._logger.error(f"Failed to send alert via {channel}: {e}")
                
                # Fire callbacks
                self._fire_callbacks(alert)
                
                # Publish event
                await self._event_bus.publish(AlertTriggeredEvent(alert))
                
                alerts.append(alert)
                self._logger.info(
                    f"Alert triggered: {alert.level.value} - {alert.message}"
                )
        
        return alerts
    
    def _create_alert(self, pothole: Pothole, distance: float) -> Alert:
        """
        Create an alert for a pothole.
        
        Args:
            pothole: The pothole
            distance: Distance to pothole in meters
            
        Returns:
            Alert entity
        """
        # Determine alert level based on severity and distance
        if pothole.severity.value == "HIGH" and distance < 50:
            level = AlertLevel.CRITICAL
            message = f"CRITICAL: Large pothole ahead in {int(distance)}m!"
        elif pothole.severity.value == "MEDIUM" or distance < 30:
            level = AlertLevel.WARNING
            message = f"WARNING: Pothole ahead in {int(distance)}m"
        else:
            level = AlertLevel.INFO
            message = f"INFO: Pothole detected {int(distance)}m away"
        
        return Alert(
            pothole_id=pothole.id,
            level=level,
            message=message,
            distance_meters=distance
        )
    
    def _fire_callbacks(self, alert: Alert):
        """Fire all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                self._logger.error(f"Callback error in {callback.__name__}: {e}")
    
    def get_history(
        self,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            severity: Filter by severity level (optional)
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts (most recent first)
        """
        alerts = list(self._history)
        
        if severity:
            severity_level = AlertLevel(severity.upper())
            alerts = [a for a in alerts if a.level == severity_level]
        
        # Most recent first
        alerts = list(reversed(alerts))
        
        return alerts[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        alerts = list(self._history)
        
        if not alerts:
            return {
                'total_alerts': 0,
                'by_level': {'INFO': 0, 'WARNING': 0, 'CRITICAL': 0},
                'acknowledged_count': 0
            }
        
        # Count by level
        level_counts = {'INFO': 0, 'WARNING': 0, 'CRITICAL': 0}
        for alert in alerts:
            level_counts[alert.level.value] += 1
        
        # Count acknowledged
        acknowledged = sum(1 for a in alerts if a.acknowledged)
        
        return {
            'total_alerts': len(alerts),
            'by_level': level_counts,
            'acknowledged_count': acknowledged,
            'acknowledgment_rate': acknowledged / len(alerts) if alerts else 0
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert by ID.
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if acknowledged, False if not found
        """
        for alert in self._history:
            if alert.id == alert_id:
                alert.acknowledged = True
                self._logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def clear_history(self):
        """Clear alert history"""
        self._history.clear()
        self._logger.info("Alert history cleared")
