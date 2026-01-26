"""
Accelerometer Data Processor Module

Signal processing pipeline for accelerometer data including:
- CSV/streaming data loading
- Sliding window extraction
- Signal filtering and baseline removal
- GPS coordinate handling
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Iterator, Generator
from dataclasses import dataclass
from scipy import signal

from ..utils import get_logger


@dataclass
class AccelWindow:
    """Represents a windowed segment of accelerometer data."""
    
    # Raw data
    accel_x: np.ndarray
    accel_y: np.ndarray
    accel_z: np.ndarray
    
    # Magnitude (computed)
    magnitude: np.ndarray
    
    # Timing
    timestamps: np.ndarray
    start_time: float
    end_time: float
    
    # GPS (if available)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    speed: Optional[float] = None
    
    # Gyroscope (if available)
    gyro_x: Optional[np.ndarray] = None
    gyro_y: Optional[np.ndarray] = None
    gyro_z: Optional[np.ndarray] = None
    
    @property
    def duration(self) -> float:
        """Window duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def sample_count(self) -> int:
        """Number of samples in window."""
        return len(self.accel_x)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'sample_count': self.sample_count,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'speed': self.speed
        }


class AccelerometerProcessor:
    """
    Processes accelerometer data from CSV files or streaming input.
    
    Features:
    - Sliding window extraction
    - Digital filtering (lowpass, highpass)
    - Baseline removal
    - Magnitude computation
    - GPS coordinate tracking
    
    Example:
        processor = AccelerometerProcessor(window_size=50, overlap=0.5)
        for window in processor.process_file("trip1_sensors.csv"):
            features = extractor.extract(window)
    """
    
    # Standard column mappings for sensor CSV files
    COLUMN_MAPPINGS = {
        'timestamp': ['timestamp', 'time', 'ts', 'Timestamp'],
        'accel_x': ['accelerometerX', 'accel_x', 'ax', 'AccX'],
        'accel_y': ['accelerometerY', 'accel_y', 'ay', 'AccY'],
        'accel_z': ['accelerometerZ', 'accel_z', 'az', 'AccZ'],
        'latitude': ['latitude', 'lat', 'Latitude'],
        'longitude': ['longitude', 'lon', 'lng', 'Longitude'],
        'speed': ['speed', 'velocity', 'Speed'],
        'gyro_x': ['gyroX', 'gyro_x', 'gx', 'GyroX'],
        'gyro_y': ['gyroY', 'gyro_y', 'gy', 'GyroY'],
        'gyro_z': ['gyroZ', 'gyro_z', 'gz', 'GyroZ'],
    }
    
    def __init__(
        self,
        window_size: int = 50,
        overlap_ratio: float = 0.5,
        sample_rate: float = 50.0,
        apply_filter: bool = True,
        filter_cutoff: float = 10.0,
        filter_order: int = 4,
        remove_baseline: bool = True
    ):
        """
        Initialize the processor.
        
        Args:
            window_size: Number of samples per window
            overlap_ratio: Window overlap ratio (0.0 to 0.9)
            sample_rate: Expected sample rate in Hz
            apply_filter: Apply lowpass filter
            filter_cutoff: Lowpass filter cutoff frequency (Hz)
            filter_order: Filter order
            remove_baseline: Remove DC offset from signals
        """
        self.logger = get_logger("accel.processor")
        
        self.window_size = window_size
        self.overlap_ratio = min(0.9, max(0.0, overlap_ratio))
        self.step_size = int(window_size * (1 - self.overlap_ratio))
        self.sample_rate = sample_rate
        
        self.apply_filter = apply_filter
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.remove_baseline = remove_baseline
        
        # Design lowpass filter
        if self.apply_filter:
            nyquist = sample_rate / 2
            normalized_cutoff = filter_cutoff / nyquist
            self.b, self.a = signal.butter(
                filter_order, 
                normalized_cutoff, 
                btype='low'
            )
        
        self.logger.info(
            f"Initialized processor: window={window_size}, "
            f"overlap={overlap_ratio}, rate={sample_rate}Hz"
        )
    
    def _find_column(self, df: pd.DataFrame, field: str) -> Optional[str]:
        """Find matching column name in DataFrame."""
        candidates = self.COLUMN_MAPPINGS.get(field, [field])
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    def load_csv(
        self,
        filepath: str,
        pothole_timestamps: Optional[List[float]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Load accelerometer CSV file.
        
        Args:
            filepath: Path to CSV file
            pothole_timestamps: Known pothole timestamps for labeling
            
        Returns:
            Tuple of (DataFrame, column mapping dict)
        """
        df = pd.read_csv(filepath)
        
        # Find column mappings
        col_map = {}
        for field in self.COLUMN_MAPPINGS:
            col = self._find_column(df, field)
            if col:
                col_map[field] = col
        
        # Verify required columns
        required = ['timestamp', 'accel_x', 'accel_y', 'accel_z']
        missing = [f for f in required if f not in col_map]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.logger.info(
            f"Loaded {len(df)} samples from {filepath}"
        )
        
        return df, col_map
    
    def _filter_signal(self, data: np.ndarray) -> np.ndarray:
        """Apply lowpass filter to signal."""
        if not self.apply_filter:
            return data
        return signal.filtfilt(self.b, self.a, data)
    
    def _remove_baseline(self, data: np.ndarray) -> np.ndarray:
        """Remove baseline (DC offset) from signal."""
        if not self.remove_baseline:
            return data
        return data - np.mean(data)
    
    def _compute_magnitude(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> np.ndarray:
        """Compute 3D magnitude of acceleration."""
        return np.sqrt(x**2 + y**2 + z**2)
    
    def process_file(
        self,
        filepath: str,
        pothole_timestamps: Optional[List[float]] = None
    ) -> Generator[AccelWindow, None, None]:
        """
        Process CSV file and yield windows.
        
        Args:
            filepath: Path to sensor CSV file
            pothole_timestamps: Known pothole timestamps (optional)
            
        Yields:
            AccelWindow objects
        """
        df, col_map = self.load_csv(filepath, pothole_timestamps)
        
        # Extract columns
        timestamps = df[col_map['timestamp']].values
        accel_x = df[col_map['accel_x']].values.astype(np.float64)
        accel_y = df[col_map['accel_y']].values.astype(np.float64)
        accel_z = df[col_map['accel_z']].values.astype(np.float64)
        
        # Optional columns
        has_gps = 'latitude' in col_map and 'longitude' in col_map
        has_speed = 'speed' in col_map
        has_gyro = all(f'gyro_{ax}' in col_map for ax in ['x', 'y', 'z'])
        
        if has_gps:
            latitudes = df[col_map['latitude']].values
            longitudes = df[col_map['longitude']].values
        
        if has_speed:
            speeds = df[col_map['speed']].values
        
        if has_gyro:
            gyro_x = df[col_map['gyro_x']].values
            gyro_y = df[col_map['gyro_y']].values
            gyro_z = df[col_map['gyro_z']].values
        
        # Apply filtering to full signals
        accel_x = self._filter_signal(accel_x)
        accel_y = self._filter_signal(accel_y)
        accel_z = self._filter_signal(accel_z)
        
        # Generate windows
        n_samples = len(timestamps)
        window_count = 0
        
        for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            
            # Extract window data
            win_x = accel_x[start_idx:end_idx].copy()
            win_y = accel_y[start_idx:end_idx].copy()
            win_z = accel_z[start_idx:end_idx].copy()
            win_ts = timestamps[start_idx:end_idx].copy()
            
            # Remove baseline per window
            if self.remove_baseline:
                win_x = self._remove_baseline(win_x)
                win_y = self._remove_baseline(win_y)
                win_z = self._remove_baseline(win_z)
            
            # Compute magnitude
            magnitude = self._compute_magnitude(win_x, win_y, win_z)
            
            # Get center GPS coordinates
            center_idx = start_idx + self.window_size // 2
            
            window = AccelWindow(
                accel_x=win_x,
                accel_y=win_y,
                accel_z=win_z,
                magnitude=magnitude,
                timestamps=win_ts,
                start_time=float(win_ts[0]),
                end_time=float(win_ts[-1]),
                latitude=float(latitudes[center_idx]) if has_gps else None,
                longitude=float(longitudes[center_idx]) if has_gps else None,
                speed=float(speeds[center_idx]) if has_speed else None,
                gyro_x=gyro_x[start_idx:end_idx] if has_gyro else None,
                gyro_y=gyro_y[start_idx:end_idx] if has_gyro else None,
                gyro_z=gyro_z[start_idx:end_idx] if has_gyro else None,
            )
            
            window_count += 1
            yield window
        
        self.logger.info(f"Generated {window_count} windows")
    
    def process_array(
        self,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> List[AccelWindow]:
        """
        Process numpy arrays directly.
        
        Args:
            accel_x: X-axis acceleration array
            accel_y: Y-axis acceleration array
            accel_z: Z-axis acceleration array
            timestamps: Optional timestamp array
            
        Returns:
            List of AccelWindow objects
        """
        # Generate synthetic timestamps if not provided
        if timestamps is None:
            timestamps = np.arange(len(accel_x)) / self.sample_rate
        
        # Apply filtering
        accel_x = self._filter_signal(accel_x.astype(np.float64))
        accel_y = self._filter_signal(accel_y.astype(np.float64))
        accel_z = self._filter_signal(accel_z.astype(np.float64))
        
        windows = []
        n_samples = len(timestamps)
        
        for start_idx in range(0, n_samples - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            
            win_x = accel_x[start_idx:end_idx].copy()
            win_y = accel_y[start_idx:end_idx].copy()
            win_z = accel_z[start_idx:end_idx].copy()
            win_ts = timestamps[start_idx:end_idx].copy()
            
            if self.remove_baseline:
                win_x = self._remove_baseline(win_x)
                win_y = self._remove_baseline(win_y)
                win_z = self._remove_baseline(win_z)
            
            magnitude = self._compute_magnitude(win_x, win_y, win_z)
            
            window = AccelWindow(
                accel_x=win_x,
                accel_y=win_y,
                accel_z=win_z,
                magnitude=magnitude,
                timestamps=win_ts,
                start_time=float(win_ts[0]),
                end_time=float(win_ts[-1])
            )
            windows.append(window)
        
        return windows
    
    def find_pothole_windows(
        self,
        windows: List[AccelWindow],
        pothole_timestamps: List[float],
        tolerance_sec: float = 0.5
    ) -> List[Tuple[AccelWindow, bool]]:
        """
        Label windows based on known pothole timestamps.
        
        Args:
            windows: List of AccelWindow objects
            pothole_timestamps: Known pothole timestamps
            tolerance_sec: Time tolerance for matching
            
        Returns:
            List of (window, is_pothole) tuples
        """
        labeled = []
        
        for window in windows:
            is_pothole = False
            for ts in pothole_timestamps:
                if window.start_time - tolerance_sec <= ts <= window.end_time + tolerance_sec:
                    is_pothole = True
                    break
            labeled.append((window, is_pothole))
        
        n_pothole = sum(1 for _, is_ph in labeled if is_ph)
        self.logger.info(
            f"Labeled {n_pothole}/{len(labeled)} windows as potholes"
        )
        
        return labeled
