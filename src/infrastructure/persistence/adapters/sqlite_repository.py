"""
SQLite Repository Implementation
"""
import sqlite3
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import json

from ..interfaces.repository_interface import RepositoryInterface
from ....domain.entities.pothole import Pothole, Severity
from ....domain.entities.alert import Alert, AlertLevel


class SQLiteRepository(RepositoryInterface):
    """
    SQLite implementation of repository interface.
    Stores potholes and alerts in SQLite database.
    """
    
    def __init__(self, database_path: str = "data/database/potholes.db"):
        """
        Initialize SQLite repository.
        
        Args:
            database_path: Path to SQLite database file
        """
        self.logger = logging.getLogger(__name__)
        self.database_path = Path(database_path)
        self.connection: Optional[sqlite3.Connection] = None
        
        # Create database directory if needed
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            self.connection = sqlite3.connect(str(self.database_path))
            self.connection.row_factory = sqlite3.Row
            
            # Create tables
            self._create_tables()
            
            self.logger.info(f"Database initialized: {self.database_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.connection.cursor()
        
        # Potholes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS potholes (
                id TEXT PRIMARY KEY,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                accel_peak REAL NOT NULL,
                bbox_area INTEGER NOT NULL,
                image_path TEXT,
                detected_at TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                pothole_id TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                distance_meters REAL NOT NULL,
                created_at TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0,
                FOREIGN KEY (pothole_id) REFERENCES potholes(id)
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_potholes_location 
            ON potholes(latitude, longitude)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_potholes_detected_at 
            ON potholes(detected_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_pothole_id 
            ON alerts(pothole_id)
        """)
        
        self.connection.commit()
        self.logger.info("Database tables created")
    
    def save_pothole(self, pothole: Pothole) -> bool:
        """Save a pothole to the database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO potholes 
                (id, latitude, longitude, severity, confidence, accel_peak, 
                 bbox_area, image_path, detected_at, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pothole.id,
                pothole.latitude,
                pothole.longitude,
                pothole.severity.value,
                pothole.confidence,
                pothole.accel_peak,
                pothole.bbox_area,
                pothole.image_path,
                pothole.detected_at.isoformat(),
                1 if pothole.is_verified else 0
            ))
            self.connection.commit()
            self.logger.debug(f"Saved pothole: {pothole.id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save pothole: {e}")
            return False
    
    def get_pothole(self, pothole_id: str) -> Optional[Pothole]:
        """Get a pothole by ID"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM potholes WHERE id = ?", (pothole_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_pothole(row)
            return None
        except Exception as e:
            self.logger.error(f"Failed to get pothole: {e}")
            return None
    
    def get_all_potholes(self) -> List[Pothole]:
        """Get all potholes"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM potholes ORDER BY detected_at DESC")
            rows = cursor.fetchall()
            
            return [self._row_to_pothole(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to get all potholes: {e}")
            return []
    
    def get_potholes_by_location(
        self,
        lat: float,
        lon: float,
        radius_m: float
    ) -> List[Pothole]:
        """
        Get potholes within a radius.
        Uses simple bounding box for efficiency.
        """
        try:
            # Approximate degrees per meter
            lat_deg_per_m = 1 / 111000
            lon_deg_per_m = 1 / (111000 * abs(lat))
            
            lat_delta = radius_m * lat_deg_per_m
            lon_delta = radius_m * lon_deg_per_m
            
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM potholes 
                WHERE latitude BETWEEN ? AND ?
                AND longitude BETWEEN ? AND ?
            """, (
                lat - lat_delta, lat + lat_delta,
                lon - lon_delta, lon + lon_delta
            ))
            rows = cursor.fetchall()
            
            potholes = [self._row_to_pothole(row) for row in rows]
            
            # Filter by actual distance
            from ....domain.services.proximity_calculator import ProximityCalculator
            calc = ProximityCalculator()
            
            return [
                p for p in potholes
                if calc.haversine_distance(lat, lon, p.latitude, p.longitude) <= radius_m
            ]
        except Exception as e:
            self.logger.error(f"Failed to get potholes by location: {e}")
            return []
    
    def save_alert(self, alert: Alert) -> bool:
        """Save an alert to the database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO alerts 
                (id, pothole_id, level, message, distance_meters, created_at, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.pothole_id,
                alert.level.value,
                alert.message,
                alert.distance_meters,
                alert.created_at.isoformat(),
                1 if alert.acknowledged else 0
            ))
            self.connection.commit()
            self.logger.debug(f"Saved alert: {alert.id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save alert: {e}")
            return False
    
    def get_alerts_by_pothole(self, pothole_id: str) -> List[Alert]:
        """Get all alerts for a pothole"""
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT * FROM alerts WHERE pothole_id = ? ORDER BY created_at DESC",
                (pothole_id,)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_alert(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to get alerts: {e}")
            return []
    
    def _row_to_pothole(self, row: sqlite3.Row) -> Pothole:
        """Convert database row to Pothole entity"""
        return Pothole(
            id=row['id'],
            latitude=row['latitude'],
            longitude=row['longitude'],
            severity=Severity(row['severity']),
            confidence=row['confidence'],
            accel_peak=row['accel_peak'],
            bbox_area=row['bbox_area'],
            image_path=row['image_path'],
            detected_at=datetime.fromisoformat(row['detected_at']),
            is_verified=bool(row['is_verified'])
        )
    
    def _row_to_alert(self, row: sqlite3.Row) -> Alert:
        """Convert database row to Alert entity"""
        return Alert(
            id=row['id'],
            pothole_id=row['pothole_id'],
            level=AlertLevel(row['level']),
            message=row['message'],
            distance_meters=row['distance_meters'],
            created_at=datetime.fromisoformat(row['created_at']),
            acknowledged=bool(row['acknowledged'])
        )
    
    def cleanup(self) -> None:
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")
