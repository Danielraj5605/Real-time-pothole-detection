"""
Accelerometer Processing Module for Pothole Detection System

Handles:
- Reading accelerometer sensor data (real hardware or CSV simulation)
- Feature extraction (peak, RMS, peak-to-peak, etc.)
- Severity classification based on accelerometer features
- Integration with vision system via fusion

Sensor Data Format (CSV):
    timestamp, latitude, longitude, speed, accelerometerX, accelerometerY, accelerometerZ, gyroX, gyroY, gyroZ
"""

import numpy as np
import pandas as pd
import logging
import json
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AccelConfig:
    """Accelerometer configuration"""
    # Sensor
    sample_rate_hz: int = 50
    
    # Windowing
    window_size_samples: int = 50
    window_overlap_ratio: float = 0.5
    
    # Signal processing
    apply_lowpass_filter: bool = True
    lowpass_cutoff_hz: float = 10.0
    remove_baseline: bool = True
    
    # Severity thresholds (from config.json)
    low_max_peak_g: float = 0.5
    low_max_rms_g: float = 0.15
    medium_max_peak_g: float = 1.5
    medium_max_rms_g: float = 0.5
    high_min_peak_g: float = 1.5
    high_min_rms_g: float = 0.5
    
    # Fusion weights
    vision_weight: float = 0.6
    accel_weight: float = 0.4
    
    # Data source
    simulation_mode: bool = True
    sensor_csv_path: str = ""
    
    @classmethod
    def from_config_json(cls, config_path: str = "config/config.json") -> 'AccelConfig':
        """Load config from config.json"""
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            
            accel_cfg = cfg.get('accelerometer', {})
            fusion_cfg = cfg.get('fusion', {}).get('rule_based', {})
            severity = accel_cfg.get('severity_thresholds', {})
            
            return cls(
                sample_rate_hz=accel_cfg.get('sensor', {}).get('sample_rate_hz', 50),
                window_size_samples=accel_cfg.get('windowing', {}).get('window_size_samples', 50),
                window_overlap_ratio=accel_cfg.get('windowing', {}).get('window_overlap_ratio', 0.5),
                apply_lowpass_filter=accel_cfg.get('signal_processing', {}).get('apply_lowpass_filter', True),
                lowpass_cutoff_hz=accel_cfg.get('signal_processing', {}).get('lowpass_cutoff_hz', 10.0),
                remove_baseline=accel_cfg.get('signal_processing', {}).get('remove_baseline', True),
                low_max_peak_g=severity.get('low', {}).get('max_peak_g', 0.5),
                low_max_rms_g=severity.get('low', {}).get('max_rms_g', 0.15),
                medium_max_peak_g=severity.get('medium', {}).get('max_peak_g', 1.5),
                medium_max_rms_g=severity.get('medium', {}).get('max_rms_g', 0.5),
                high_min_peak_g=severity.get('high', {}).get('min_peak_g', 1.5),
                high_min_rms_g=severity.get('high', {}).get('min_rms_g', 0.5),
                vision_weight=fusion_cfg.get('vision_weight', 0.6),
                accel_weight=fusion_cfg.get('accel_weight', 0.4),
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not load config: {e}, using defaults")
            return cls()


# =============================================================================
# ACCELEROMETER DATA
# =============================================================================

@dataclass
class AccelFeatures:
    """Extracted accelerometer features for a time window"""
    timestamp: float = 0.0
    peak_acceleration: float = 0.0
    rms_vibration: float = 0.0
    peak_to_peak: float = 0.0
    mean_acceleration: float = 0.0
    std_acceleration: float = 0.0
    crest_factor: float = 0.0
    zero_crossing_rate: float = 0.0
    accel_severity: str = "NONE"

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'peak_acceleration': self.peak_acceleration,
            'rms_vibration': self.rms_vibration,
            'peak_to_peak': self.peak_to_peak,
            'mean_acceleration': self.mean_acceleration,
            'std_acceleration': self.std_acceleration,
            'crest_factor': self.crest_factor,
            'zero_crossing_rate': self.zero_crossing_rate,
            'accel_severity': self.accel_severity,
        }


# =============================================================================
# ACCELEROMETER PROCESSOR
# =============================================================================

class AccelerometerProcessor:
    """
    Processes accelerometer data for pothole detection.
    
    Supports two modes:
    1. Simulation mode: reads from CSV files (for testing)
    2. Live mode: reads from a real-time buffer (for deployment)
    """
    
    def __init__(self, config: Optional[AccelConfig] = None):
        self.config = config or AccelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Live data buffer
        self._buffer: deque = deque(maxlen=self.config.window_size_samples * 2)
        
        # Simulation data
        self._sim_data: Optional[pd.DataFrame] = None
        self._sim_index: int = 0
        
        # State
        self._is_initialized = False
        self._latest_features: Optional[AccelFeatures] = None
    
    def initialize(self, csv_path: str = "") -> bool:
        """Initialize the processor"""
        try:
            if csv_path or self.config.sensor_csv_path:
                path = csv_path or self.config.sensor_csv_path
                self._load_csv(path)
                self.config.simulation_mode = True
            
            self._is_initialized = True
            self.logger.info("[OK] Accelerometer processor initialized"
                           + (" (simulation mode)" if self.config.simulation_mode else " (live mode)"))
            return True
        except Exception as e:
            self.logger.error(f"Accelerometer init failed: {e}")
            return False
    
    def _load_csv(self, path: str):
        """Load CSV sensor data for simulation"""
        self._sim_data = pd.read_csv(path)
        self._sim_index = 0
        self.logger.info(f"  Loaded {len(self._sim_data)} sensor samples from {Path(path).name}")
    
    # -------------------------------------------------------------------------
    # Data Acquisition
    # -------------------------------------------------------------------------
    
    def push_sample(self, accel_x: float, accel_y: float, accel_z: float,
                    timestamp: float = None):
        """Push a single accelerometer sample (for live mode)"""
        self._buffer.append({
            'timestamp': timestamp or time.time(),
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
        })
    
    def get_current_window(self) -> Optional[np.ndarray]:
        """Get current window of accelerometer data"""
        ws = self.config.window_size_samples
        
        if self.config.simulation_mode and self._sim_data is not None:
            if self._sim_index + ws > len(self._sim_data):
                return None
            
            window = self._sim_data.iloc[self._sim_index:self._sim_index + ws]
            step = int(ws * (1 - self.config.window_overlap_ratio))
            self._sim_index += max(step, 1)
            
            return np.column_stack([
                window['accelerometerX'].values,
                window['accelerometerY'].values,
                window['accelerometerZ'].values,
            ])
        else:
            if len(self._buffer) < ws:
                return None
            
            samples = list(self._buffer)[-ws:]
            return np.array([[s['accel_x'], s['accel_y'], s['accel_z']] for s in samples])
    
    def get_sim_timestamp(self) -> float:
        """Get current simulation timestamp"""
        if self._sim_data is not None and self._sim_index < len(self._sim_data):
            return self._sim_data.iloc[self._sim_index]['timestamp']
        return time.time()
    
    # -------------------------------------------------------------------------
    # Feature Extraction
    # -------------------------------------------------------------------------
    
    def extract_features(self, window: np.ndarray = None) -> Optional[AccelFeatures]:
        """
        Extract features from accelerometer window.
        
        Args:
            window: Nx3 array [accel_x, accel_y, accel_z]. If None, uses current window.
        
        Returns:
            AccelFeatures or None
        """
        if window is None:
            window = self.get_current_window()
        
        if window is None or len(window) < 5:
            return None
        
        accel_y = window[:, 1]  # Vertical axis (most relevant for potholes)
        accel_z = window[:, 2]  # Forward axis
        
        # Remove gravity baseline if configured
        if self.config.remove_baseline:
            accel_y = accel_y - np.mean(accel_y)
            accel_z = accel_z - np.mean(accel_z)
        
        # Combined vertical magnitude
        magnitude = np.sqrt(accel_y**2 + accel_z**2)
        
        # Extract features
        peak = np.max(np.abs(magnitude))
        rms = np.sqrt(np.mean(magnitude**2))
        p2p = np.max(magnitude) - np.min(magnitude)
        mean_val = np.mean(magnitude)
        std_val = np.std(magnitude)
        crest = peak / rms if rms > 0 else 0
        
        # Zero crossing rate
        signs = np.sign(accel_y)
        zcr = np.sum(np.abs(np.diff(signs)) > 0) / len(accel_y)
        
        features = AccelFeatures(
            timestamp=time.time(),
            peak_acceleration=float(peak),
            rms_vibration=float(rms),
            peak_to_peak=float(p2p),
            mean_acceleration=float(mean_val),
            std_acceleration=float(std_val),
            crest_factor=float(crest),
            zero_crossing_rate=float(zcr),
        )
        
        # Classify severity from accelerometer alone
        features.accel_severity = self._classify_accel_severity(features)
        
        self._latest_features = features
        return features
    
    def _classify_accel_severity(self, features: AccelFeatures) -> str:
        """Classify severity based on accelerometer features only"""
        peak = features.peak_acceleration
        rms = features.rms_vibration
        
        if peak >= self.config.high_min_peak_g or rms >= self.config.high_min_rms_g:
            return "HIGH"
        elif peak >= self.config.low_max_peak_g or rms >= self.config.low_max_rms_g:
            return "MEDIUM"
        elif peak > 0.1:  # Above noise floor
            return "LOW"
        else:
            return "NONE"
    
    # -------------------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------------------
    
    def fuse_severity(self, vision_confidence: float, vision_area_ratio: float,
                      accel_features: Optional[AccelFeatures] = None) -> Dict:
        """
        Fuse vision and accelerometer data for final severity classification.
        
        Args:
            vision_confidence: YOLO detection confidence (0-1)
            vision_area_ratio: Bounding box area / frame area
            accel_features: Accelerometer features (uses latest if None)
        
        Returns:
            Dict with 'severity', 'depth', 'fusion_score', 'accel_severity'
        """
        if accel_features is None:
            accel_features = self._latest_features
        
        vw = self.config.vision_weight
        aw = self.config.accel_weight
        
        # --- Vision score (0-1) ---
        vision_score = vision_confidence * min(vision_area_ratio / 0.15, 1.0)
        
        # --- Accelerometer score (0-1) ---
        if accel_features and accel_features.accel_severity != "NONE":
            accel_score = min(accel_features.peak_acceleration / 3.0, 1.0)
            has_accel = True
        else:
            accel_score = 0.0
            has_accel = False
        
        # --- Weighted fusion ---
        if has_accel:
            fusion_score = vw * vision_score + aw * accel_score
        else:
            # Fall back to vision-only when no accel data available
            fusion_score = vision_score
        
        # --- Severity classification ---
        # Rule 1: Strong accelerometer signal overrides
        if has_accel and accel_features.peak_acceleration >= 2.0:
            severity = "HIGH"
        # Rule 2: Large visual + strong accel
        elif has_accel and vision_confidence > 0.8 and vision_area_ratio > 0.1 and accel_features.peak_acceleration > 0.5:
            severity = "HIGH"
        # Rule 3: Fusion score
        elif fusion_score > 0.7:
            severity = "HIGH"
        elif fusion_score > 0.4:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # --- Depth estimate ---
        if vision_area_ratio > 0.15 or (has_accel and accel_features.peak_acceleration > 2.0):
            depth = "DEEP"
        elif vision_area_ratio > 0.08 or (has_accel and accel_features.peak_acceleration > 1.0):
            depth = "MODERATE"
        else:
            depth = "SHALLOW"
        
        return {
            'severity': severity,
            'depth': depth,
            'fusion_score': round(fusion_score, 3),
            'vision_score': round(vision_score, 3),
            'accel_score': round(accel_score, 3),
            'accel_severity': accel_features.accel_severity if accel_features else "NONE",
            'accel_peak_g': accel_features.peak_acceleration if accel_features else 0.0,
            'accel_rms_g': accel_features.rms_vibration if accel_features else 0.0,
            'has_accel_data': has_accel,
        }
    
    @property
    def latest_features(self) -> Optional[AccelFeatures]:
        return self._latest_features
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
