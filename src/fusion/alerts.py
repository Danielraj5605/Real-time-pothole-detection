"""
Alert Manager Module

Manages pothole detection alerts with debouncing and logging.
"""

import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import math

from .engine import FusionResult
from ..utils import get_logger


@dataclass
class Alert:
    """Represents a single pothole alert."""
    id: int
    timestamp: str
    severity: str
    confidence: float
    latitude: Optional[float]
    longitude: Optional[float]
    vision_confidence: float
    accel_peak: float
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'severity': self.severity,
            'confidence': self.confidence,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'vision_confidence': self.vision_confidence,
            'accel_peak': self.accel_peak,
            'acknowledged': self.acknowledged
        }
    
    def __str__(self) -> str:
        loc = f"({self.latitude:.4f}, {self.longitude:.4f})" if self.latitude else "unknown"
        return (
            f"Alert #{self.id}: {self.severity.upper()} pothole "
            f"@ {loc} (conf={self.confidence:.0%})"
        )


class AlertManager:
    """
    Manages pothole alerts with debouncing and callbacks.
    
    Features:
    - Debouncing: Prevents duplicate alerts for the same pothole
    - Callbacks: Register handlers for new alerts
    - History: Maintains alert history
    - Severity filtering: Filter by minimum severity
    
    Example:
        manager = AlertManager(debounce_seconds=2.0)
        manager.add_callback(lambda alert: print(alert))
        
        # In detection loop
        if fusion_result.pothole_detected:
            manager.process(fusion_result)
    """
    
    SEVERITY_LEVELS = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    
    def __init__(
        self,
        debounce_seconds: float = 2.0,
        min_distance_meters: float = 5.0,
        min_severity: str = 'low',
        max_history: int = 100
    ):
        """
        Initialize alert manager.
        
        Args:
            debounce_seconds: Minimum time between alerts
            min_distance_meters: Minimum distance between alerts (GPS)
            min_severity: Minimum severity to generate alert
            max_history: Maximum alerts to keep in history
        """
        self.logger = get_logger("fusion.alerts")
        self.debounce_seconds = debounce_seconds
        self.min_distance_meters = min_distance_meters
        self.min_severity_level = self.SEVERITY_LEVELS.get(min_severity, 1)
        self.max_history = max_history
        
        # State
        self._next_id = 1
        self._last_alert_time: float = 0
        self._last_alert_location: Optional[tuple] = None
        self._history: deque = deque(maxlen=max_history)
        self._callbacks: List[Callable[[Alert], None]] = []
        
        self.logger.info(
            f"AlertManager initialized: debounce={debounce_seconds}s, "
            f"min_severity={min_severity}"
        )
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Register a callback for new alerts."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Alert], None]) -> bool:
        """Remove a callback."""
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False
    
    def process(self, result: FusionResult) -> Optional[Alert]:
        """
        Process fusion result and potentially generate alert.
        
        Args:
            result: FusionResult from fusion engine
            
        Returns:
            Alert if generated, None otherwise
        """
        if not result.pothole_detected:
            return None
        
        # Check severity threshold
        severity_level = self.SEVERITY_LEVELS.get(result.severity, 0)
        if severity_level < self.min_severity_level:
            return None
        
        # Check debounce
        current_time = time.time()
        if self._should_debounce(result, current_time):
            self.logger.debug("Alert debounced")
            return None
        
        # Create alert
        alert = Alert(
            id=self._next_id,
            timestamp=result.timestamp,
            severity=result.severity,
            confidence=result.confidence,
            latitude=result.latitude,
            longitude=result.longitude,
            vision_confidence=result.vision_confidence,
            accel_peak=result.accel_peak
        )
        
        self._next_id += 1
        self._last_alert_time = current_time
        if result.latitude and result.longitude:
            self._last_alert_location = (result.latitude, result.longitude)
        
        # Store in history
        self._history.append(alert)
        
        # Fire callbacks
        self._fire_callbacks(alert)
        
        self.logger.info(str(alert))
        
        return alert
    
    def _should_debounce(
        self,
        result: FusionResult,
        current_time: float
    ) -> bool:
        """Check if alert should be debounced."""
        # Time-based debounce
        if current_time - self._last_alert_time < self.debounce_seconds:
            return True
        
        # Distance-based debounce (if GPS available)
        if (result.latitude and result.longitude and 
            self._last_alert_location):
            distance = self._haversine_distance(
                self._last_alert_location[0],
                self._last_alert_location[1],
                result.latitude,
                result.longitude
            )
            if distance < self.min_distance_meters:
                return True
        
        return False
    
    def _haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two GPS points in meters."""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _fire_callbacks(self, alert: Alert):
        """Fire all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def get_history(
        self,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            severity: Filter by severity (optional)
            limit: Maximum number of alerts to return
            
        Returns:
            List of Alert objects (most recent first)
        """
        alerts = list(self._history)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Most recent first
        alerts = list(reversed(alerts))
        
        return alerts[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        alerts = list(self._history)
        
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        for alert in alerts:
            if alert.severity in severity_counts:
                severity_counts[alert.severity] += 1
        
        return {
            'total_alerts': len(alerts),
            'severity_counts': severity_counts,
            'avg_confidence': sum(a.confidence for a in alerts) / max(len(alerts), 1),
            'avg_peak': sum(a.accel_peak for a in alerts) / max(len(alerts), 1)
        }
    
    def acknowledge(self, alert_id: int) -> bool:
        """Acknowledge an alert by ID."""
        for alert in self._history:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def clear_history(self):
        """Clear alert history."""
        self._history.clear()
        self.logger.info("Alert history cleared")
