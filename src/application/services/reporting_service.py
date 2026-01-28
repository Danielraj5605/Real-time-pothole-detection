"""
Reporting Service - Generates Reports and Statistics
"""
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
from collections import Counter

from ...domain.entities.pothole import Pothole, Severity


class ReportingService:
    """
    Service for generating reports and statistics.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def generate_summary(self, potholes: List[Pothole]) -> Dict[str, Any]:
        """
        Generate summary statistics for potholes.
        
        Args:
            potholes: List of detected potholes
            
        Returns:
            Dictionary with summary statistics
        """
        if not potholes:
            return {
                'total_count': 0,
                'by_severity': {},
                'average_confidence': 0.0,
                'verified_count': 0
            }
        
        # Count by severity
        severity_counts = Counter(p.severity for p in potholes)
        
        # Calculate averages
        avg_confidence = sum(p.confidence for p in potholes) / len(potholes)
        avg_accel = sum(p.accel_peak for p in potholes) / len(potholes)
        
        # Verified count
        verified = sum(1 for p in potholes if p.is_verified)
        
        return {
            'total_count': len(potholes),
            'by_severity': {
                'HIGH': severity_counts.get(Severity.HIGH, 0),
                'MEDIUM': severity_counts.get(Severity.MEDIUM, 0),
                'LOW': severity_counts.get(Severity.LOW, 0)
            },
            'average_confidence': round(avg_confidence, 3),
            'average_acceleration': round(avg_accel, 3),
            'verified_count': verified,
            'verification_rate': round(verified / len(potholes), 3) if potholes else 0
        }
    
    def generate_time_series(
        self,
        potholes: List[Pothole],
        interval_hours: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate time series data for potholes.
        
        Args:
            potholes: List of potholes
            interval_hours: Time interval in hours
            
        Returns:
            List of time buckets with counts
        """
        if not potholes:
            return []
        
        # Sort by detection time
        sorted_potholes = sorted(potholes, key=lambda p: p.detected_at)
        
        # Create time buckets
        start_time = sorted_potholes[0].detected_at
        end_time = sorted_potholes[-1].detected_at
        
        buckets = []
        current_time = start_time
        interval = timedelta(hours=interval_hours)
        
        while current_time <= end_time:
            next_time = current_time + interval
            
            # Count potholes in this bucket
            count = sum(
                1 for p in sorted_potholes
                if current_time <= p.detected_at < next_time
            )
            
            buckets.append({
                'timestamp': current_time.isoformat(),
                'count': count
            })
            
            current_time = next_time
        
        return buckets
    
    def generate_geographic_report(
        self,
        potholes: List[Pothole],
        grid_size_deg: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Generate geographic distribution report.
        
        Args:
            potholes: List of potholes
            grid_size_deg: Grid cell size in degrees
            
        Returns:
            List of grid cells with pothole counts
        """
        if not potholes:
            return []
        
        # Group by grid cell
        grid_counts = {}
        
        for pothole in potholes:
            # Calculate grid cell
            lat_cell = int(pothole.latitude / grid_size_deg)
            lon_cell = int(pothole.longitude / grid_size_deg)
            cell_key = (lat_cell, lon_cell)
            
            if cell_key not in grid_counts:
                grid_counts[cell_key] = {
                    'lat_min': lat_cell * grid_size_deg,
                    'lat_max': (lat_cell + 1) * grid_size_deg,
                    'lon_min': lon_cell * grid_size_deg,
                    'lon_max': (lon_cell + 1) * grid_size_deg,
                    'count': 0,
                    'severities': Counter()
                }
            
            grid_counts[cell_key]['count'] += 1
            grid_counts[cell_key]['severities'][pothole.severity] += 1
        
        # Convert to list
        result = []
        for cell_data in grid_counts.values():
            result.append({
                'lat_min': cell_data['lat_min'],
                'lat_max': cell_data['lat_max'],
                'lon_min': cell_data['lon_min'],
                'lon_max': cell_data['lon_max'],
                'count': cell_data['count'],
                'high_severity': cell_data['severities'].get(Severity.HIGH, 0),
                'medium_severity': cell_data['severities'].get(Severity.MEDIUM, 0),
                'low_severity': cell_data['severities'].get(Severity.LOW, 0)
            })
        
        return result
