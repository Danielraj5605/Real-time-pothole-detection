"""
Logging Module

Provides structured logging with file rotation, console output,
and configurable log levels for the pothole detection system.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "pothole_detection",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    colored_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size_mb: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        colored_console: Whether to use colored console output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if colored_console and sys.platform != 'win32':
        # Use colored formatter for non-Windows systems
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        console_handler.setFormatter(ColoredFormatter(console_format))
    else:
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        console_handler.setFormatter(logging.Formatter(console_format))
    
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        file_handler.setFormatter(logging.Formatter(file_format))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "pothole_detection") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class EventLogger:
    """
    Specialized logger for pothole detection events.
    Logs events to SQLite database for analysis.
    """
    
    def __init__(self, db_path: str = "logs/pothole_events.db"):
        """
        Initialize event logger.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self.logger = get_logger("event_logger")
    
    def _init_database(self):
        """Initialize SQLite database with events table."""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pothole_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                detection_confidence REAL,
                severity TEXT,
                vision_confidence REAL,
                bbox_area REAL,
                accel_peak REAL,
                accel_rms REAL,
                fusion_score REAL,
                image_path TEXT,
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON pothole_events(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_severity 
            ON pothole_events(severity)
        """)
        
        conn.commit()
        conn.close()
    
    def log_event(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        detection_confidence: float = 0.0,
        severity: str = "unknown",
        vision_confidence: float = 0.0,
        bbox_area: float = 0.0,
        accel_peak: float = 0.0,
        accel_rms: float = 0.0,
        fusion_score: float = 0.0,
        image_path: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Log a pothole detection event to the database.
        
        Returns:
            Event ID
        """
        import sqlite3
        
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO pothole_events (
                timestamp, latitude, longitude, detection_confidence,
                severity, vision_confidence, bbox_area, accel_peak,
                accel_rms, fusion_score, image_path, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, latitude, longitude, detection_confidence,
            severity, vision_confidence, bbox_area, accel_peak,
            accel_rms, fusion_score, image_path, notes
        ))
        
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.logger.info(
            f"Event logged: ID={event_id}, Severity={severity}, "
            f"Confidence={detection_confidence:.2f}"
        )
        
        return event_id
    
    def get_events(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        Query events from the database.
        
        Args:
            start_time: ISO format start time filter
            end_time: ISO format end time filter
            severity: Severity level filter
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM pothole_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
