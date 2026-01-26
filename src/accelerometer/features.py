"""
Accelerometer Feature Extractor Module

Extracts time-domain and statistical features from accelerometer windows
for pothole severity classification.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .processor import AccelWindow
from ..utils import get_logger


@dataclass
class AccelFeatures:
    """Features extracted from accelerometer window for fusion."""
    
    # Peak metrics
    peak_acceleration: float  # Maximum absolute acceleration (g)
    peak_x: float
    peak_y: float
    peak_z: float
    
    # RMS vibration
    rms_vibration: float  # RMS of magnitude
    rms_x: float
    rms_y: float
    rms_z: float
    
    # Statistical features
    mean_acceleration: float
    std_acceleration: float
    peak_to_peak: float
    
    # Signal characteristics
    crest_factor: float  # Peak / RMS ratio
    zero_crossing_rate: float
    
    # Motion context
    speed: Optional[float] = None
    
    # Derived severity (from classifier or rules)
    severity: str = "unknown"
    severity_confidence: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array for ML models."""
        return np.array([
            self.peak_acceleration,
            self.peak_x,
            self.peak_y,
            self.peak_z,
            self.rms_vibration,
            self.rms_x,
            self.rms_y,
            self.rms_z,
            self.mean_acceleration,
            self.std_acceleration,
            self.peak_to_peak,
            self.crest_factor,
            self.zero_crossing_rate,
            self.speed if self.speed is not None else 0.0
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'peak_acceleration': self.peak_acceleration,
            'peak_x': self.peak_x,
            'peak_y': self.peak_y,
            'peak_z': self.peak_z,
            'rms_vibration': self.rms_vibration,
            'rms_x': self.rms_x,
            'rms_y': self.rms_y,
            'rms_z': self.rms_z,
            'mean_acceleration': self.mean_acceleration,
            'std_acceleration': self.std_acceleration,
            'peak_to_peak': self.peak_to_peak,
            'crest_factor': self.crest_factor,
            'zero_crossing_rate': self.zero_crossing_rate,
            'speed': self.speed,
            'severity': self.severity,
            'severity_confidence': self.severity_confidence
        }
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names for ML."""
        return [
            'peak_acceleration',
            'peak_x',
            'peak_y',
            'peak_z',
            'rms_vibration',
            'rms_x',
            'rms_y',
            'rms_z',
            'mean_acceleration',
            'std_acceleration',
            'peak_to_peak',
            'crest_factor',
            'zero_crossing_rate',
            'speed'
        ]


class AccelFeatureExtractor:
    """
    Extracts features from accelerometer windows for severity classification.
    
    Features are designed for:
    - Pothole vs normal road detection
    - Severity classification (low/medium/high)
    - Multimodal fusion with vision features
    
    Example:
        extractor = AccelFeatureExtractor()
        features = extractor.extract(accel_window)
        print(f"Peak: {features.peak_acceleration:.2f}g")
    """
    
    def __init__(self, gravity_offset: float = 1.0):
        """
        Initialize the feature extractor.
        
        Args:
            gravity_offset: Expected gravity component to remove (typically 1.0g)
        """
        self.logger = get_logger("accel.features")
        self.gravity_offset = gravity_offset
    
    def extract(self, window: AccelWindow) -> AccelFeatures:
        """
        Extract features from an accelerometer window.
        
        Args:
            window: AccelWindow object
            
        Returns:
            AccelFeatures dataclass
        """
        # Get signals
        x = window.accel_x
        y = window.accel_y
        z = window.accel_z
        mag = window.magnitude
        
        # Adjust magnitude for gravity (Z-axis typically has ~1g offset)
        # For vertical z-axis, remove gravity: sqrt(x^2 + y^2 + (z-1)^2)
        # But we use baseline-removed signals, so magnitude should be near 0 at rest
        
        # Peak metrics (absolute values for detecting impacts)
        peak_x = float(np.max(np.abs(x)))
        peak_y = float(np.max(np.abs(y)))
        peak_z = float(np.max(np.abs(z)))
        peak_acceleration = float(np.max(np.abs(mag)))
        
        # RMS metrics (overall vibration energy)
        rms_x = float(np.sqrt(np.mean(x ** 2)))
        rms_y = float(np.sqrt(np.mean(y ** 2)))
        rms_z = float(np.sqrt(np.mean(z ** 2)))
        rms_vibration = float(np.sqrt(np.mean(mag ** 2)))
        
        # Statistical features
        mean_acceleration = float(np.mean(mag))
        std_acceleration = float(np.std(mag))
        peak_to_peak = float(np.max(mag) - np.min(mag))
        
        # Crest factor (peak / RMS) - high for impulsive signals
        crest_factor = peak_acceleration / max(rms_vibration, 1e-6)
        
        # Zero crossing rate (for frequency content estimation)
        # Using magnitude centered on mean
        centered_mag = mag - np.mean(mag)
        zero_crossings = np.sum(np.diff(np.sign(centered_mag)) != 0)
        zero_crossing_rate = float(zero_crossings / len(mag))
        
        return AccelFeatures(
            peak_acceleration=peak_acceleration,
            peak_x=peak_x,
            peak_y=peak_y,
            peak_z=peak_z,
            rms_vibration=rms_vibration,
            rms_x=rms_x,
            rms_y=rms_y,
            rms_z=rms_z,
            mean_acceleration=mean_acceleration,
            std_acceleration=std_acceleration,
            peak_to_peak=peak_to_peak,
            crest_factor=crest_factor,
            zero_crossing_rate=zero_crossing_rate,
            speed=window.speed
        )
    
    def extract_batch(
        self,
        windows: List[AccelWindow]
    ) -> List[AccelFeatures]:
        """
        Extract features from multiple windows.
        
        Args:
            windows: List of AccelWindow objects
            
        Returns:
            List of AccelFeatures objects
        """
        return [self.extract(w) for w in windows]
    
    def to_feature_matrix(
        self,
        features_list: List[AccelFeatures]
    ) -> np.ndarray:
        """
        Convert list of features to numpy matrix for ML.
        
        Args:
            features_list: List of AccelFeatures objects
            
        Returns:
            2D numpy array (n_samples, n_features)
        """
        return np.vstack([f.to_array() for f in features_list])
    
    def compute_severity_rule_based(
        self,
        features: AccelFeatures,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ) -> tuple:
        """
        Compute severity using rule-based thresholds.
        
        Args:
            features: AccelFeatures object
            thresholds: Severity thresholds dict (uses defaults if None)
            
        Returns:
            Tuple of (severity_string, confidence)
        """
        if thresholds is None:
            thresholds = {
                'low': {'max_peak_g': 0.5, 'max_rms_g': 0.15},
                'medium': {'max_peak_g': 1.5, 'max_rms_g': 0.5},
                'high': {'min_peak_g': 1.5, 'min_rms_g': 0.5}
            }
        
        peak = features.peak_acceleration
        rms = features.rms_vibration
        
        # High severity
        if (peak >= thresholds['high']['min_peak_g'] or 
            rms >= thresholds['high']['min_rms_g']):
            confidence = min(1.0, peak / 2.0)  # Normalize to [0, 1]
            return 'high', confidence
        
        # Medium severity
        if (peak >= thresholds['low']['max_peak_g'] or
            rms >= thresholds['low']['max_rms_g']):
            confidence = 0.5 + (peak - 0.5) / 2.0  # Scale to [0.5, 1.0]
            return 'medium', min(1.0, confidence)
        
        # Low severity
        confidence = max(0.3, 1.0 - peak / 0.5)  # Higher peak = lower confidence it's low
        return 'low', confidence
    
    def is_pothole_candidate(
        self,
        features: AccelFeatures,
        peak_threshold: float = 0.3,
        rms_threshold: float = 0.1
    ) -> bool:
        """
        Quick check if window might contain a pothole.
        
        Args:
            features: Extracted features
            peak_threshold: Peak acceleration threshold (g)
            rms_threshold: RMS threshold (g)
            
        Returns:
            True if pothole candidate
        """
        return (features.peak_acceleration >= peak_threshold or
                features.rms_vibration >= rms_threshold)
