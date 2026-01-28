"""
Proximity Calculation Service
"""
from typing import List, Tuple
from math import radians, sin, cos, sqrt, atan2
from ..entities.pothole import Pothole


class ProximityCalculator:
    """
    Service for calculating proximity and determining alert triggers.
    """
    
    @staticmethod
    def haversine_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth's radius in meters
        lat1_rad, lon1_rad = radians(lat1), radians(lon1)
        lat2_rad, lon2_rad = radians(lat2), radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def find_nearby_potholes(
        self,
        current_lat: float,
        current_lon: float,
        potholes: List[Pothole],
        max_distance_m: float = 200
    ) -> List[Tuple[Pothole, float]]:
        """
        Find potholes within a certain distance.
        
        Args:
            current_lat: Current latitude
            current_lon: Current longitude
            potholes: List of known potholes
            max_distance_m: Maximum distance to consider (meters)
            
        Returns:
            List of (pothole, distance) tuples sorted by distance
        """
        nearby = []
        
        for pothole in potholes:
            distance = self.haversine_distance(
                current_lat, current_lon,
                pothole.latitude, pothole.longitude
            )
            
            if distance <= max_distance_m:
                nearby.append((pothole, distance))
        
        # Sort by distance (closest first)
        nearby.sort(key=lambda x: x[1])
        
        return nearby
    
    def should_trigger_alert(
        self,
        pothole: Pothole,
        distance_m: float
    ) -> bool:
        """
        Determine if an alert should be triggered based on pothole and distance.
        
        Args:
            pothole: The pothole to check
            distance_m: Distance to the pothole in meters
            
        Returns:
            True if alert should be triggered
        """
        return pothole.should_alert_at_distance(distance_m)
