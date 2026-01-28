"""
Repository Interface - Abstract Data Persistence
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime

from ....domain.entities.pothole import Pothole
from ....domain.entities.alert import Alert


class RepositoryInterface(ABC):
    """Abstract interface for data persistence"""
    
    @abstractmethod
    def save_pothole(self, pothole: Pothole) -> bool:
        """
        Save a pothole to the repository.
        
        Args:
            pothole: Pothole entity to save
            
        Returns:
            True if saved successfully
        """
        pass
    
    @abstractmethod
    def get_pothole(self, pothole_id: str) -> Optional[Pothole]:
        """
        Get a pothole by ID.
        
        Args:
            pothole_id: Pothole ID
            
        Returns:
            Pothole entity or None
        """
        pass
    
    @abstractmethod
    def get_all_potholes(self) -> List[Pothole]:
        """
        Get all potholes.
        
        Returns:
            List of all potholes
        """
        pass
    
    @abstractmethod
    def get_potholes_by_location(
        self,
        lat: float,
        lon: float,
        radius_m: float
    ) -> List[Pothole]:
        """
        Get potholes within a radius of a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_m: Radius in meters
            
        Returns:
            List of potholes within radius
        """
        pass
    
    @abstractmethod
    def save_alert(self, alert: Alert) -> bool:
        """
        Save an alert to the repository.
        
        Args:
            alert: Alert entity to save
            
        Returns:
            True if saved successfully
        """
        pass
    
    @abstractmethod
    def get_alerts_by_pothole(self, pothole_id: str) -> List[Alert]:
        """
        Get all alerts for a pothole.
        
        Args:
            pothole_id: Pothole ID
            
        Returns:
            List of alerts
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release repository resources"""
        pass
