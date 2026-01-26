"""
Fusion Engine Module

Core multimodal fusion engine that combines vision and accelerometer
pipelines for robust pothole detection and severity classification.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..vision.features import VisionFeatures
from ..accelerometer.features import AccelFeatures
from ..accelerometer.classifier import SeverityPrediction
from ..utils import get_logger


@dataclass
class FusionResult:
    """Complete result from multimodal fusion."""
    
    # Detection result
    pothole_detected: bool
    confidence: float
    severity: str  # 'none', 'low', 'medium', 'high'
    
    # Source contributions
    vision_detected: bool
    vision_confidence: float
    accel_detected: bool
    accel_peak: float
    accel_rms: float
    
    # Location
    latitude: Optional[float]
    longitude: Optional[float]
    
    # Metadata
    timestamp: str
    fusion_method: str
    
    # Raw features (for debugging/analysis)
    vision_features: Optional[VisionFeatures] = None
    accel_features: Optional[AccelFeatures] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pothole_detected': self.pothole_detected,
            'confidence': self.confidence,
            'severity': self.severity,
            'vision_detected': self.vision_detected,
            'vision_confidence': self.vision_confidence,
            'accel_detected': self.accel_detected,
            'accel_peak': self.accel_peak,
            'accel_rms': self.accel_rms,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'timestamp': self.timestamp,
            'fusion_method': self.fusion_method
        }
    
    def __str__(self) -> str:
        status = "DETECTED" if self.pothole_detected else "clear"
        return (
            f"FusionResult({status}, severity={self.severity}, "
            f"conf={self.confidence:.2f}, "
            f"vision={self.vision_confidence:.2f}, "
            f"peak={self.accel_peak:.2f}g)"
        )


class FusionEngine:
    """
    Multimodal fusion engine for pothole detection.
    
    Combines:
    - Vision-based detection (YOLOv8 confidence, bbox features)
    - Accelerometer-based detection (peak, RMS, severity)
    
    Fusion strategies:
    - Rule-based: Configurable thresholds and logic
    - Weighted average: Simple weighted combination
    - ML-based: Trained classifier on combined features (future)
    
    Example:
        engine = FusionEngine()
        result = engine.fuse(vision_features, accel_features)
        if result.pothole_detected:
            print(f"Pothole: {result.severity} ({result.confidence:.0%})")
    """
    
    def __init__(
        self,
        method: str = "rule_based",
        vision_weight: float = 0.6,
        accel_weight: float = 0.4,
        detection_threshold: float = 0.5,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the fusion engine.
        
        Args:
            method: Fusion method ('rule_based', 'weighted_average')
            vision_weight: Weight for vision confidence in fusion
            accel_weight: Weight for accelerometer score in fusion
            detection_threshold: Threshold for positive detection
            config: Optional configuration override
        """
        self.logger = get_logger("fusion.engine")
        self.method = method
        self.vision_weight = vision_weight
        self.accel_weight = accel_weight
        self.detection_threshold = detection_threshold
        
        # Load config or use defaults
        self.config = config or self._default_config()
        
        self.logger.info(
            f"Initialized FusionEngine: method={method}, "
            f"weights=({vision_weight:.1f}/{accel_weight:.1f})"
        )
    
    def _default_config(self) -> Dict[str, Any]:
        """Default fusion configuration."""
        return {
            'vision_confidence_threshold': 0.5,
            'accel_peak_threshold': 0.3,
            'accel_rms_threshold': 0.15,
            'severity_overrides': {
                'high_impact': {'accel_peak': 2.0, 'accel_rms': 0.8},
                'large_visual': {'bbox_area': 0.1, 'vision_conf': 0.8}
            }
        }
    
    def fuse(
        self,
        vision_features: Optional[VisionFeatures],
        accel_features: Optional[AccelFeatures],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        accel_severity: Optional[SeverityPrediction] = None
    ) -> FusionResult:
        """
        Perform multimodal fusion.
        
        Args:
            vision_features: Features from vision pipeline (optional)
            accel_features: Features from accelerometer pipeline (optional)
            latitude: GPS latitude (optional)
            longitude: GPS longitude (optional)
            accel_severity: Pre-computed severity prediction (optional)
            
        Returns:
            FusionResult object
        """
        timestamp = datetime.now().isoformat()
        
        # Handle missing inputs
        if vision_features is None and accel_features is None:
            return FusionResult(
                pothole_detected=False,
                confidence=0.0,
                severity='none',
                vision_detected=False,
                vision_confidence=0.0,
                accel_detected=False,
                accel_peak=0.0,
                accel_rms=0.0,
                latitude=latitude,
                longitude=longitude,
                timestamp=timestamp,
                fusion_method=self.method
            )
        
        # Extract values with defaults
        vision_conf = vision_features.confidence if vision_features else 0.0
        vision_detected = (vision_features.detected if vision_features 
                          else False)
        
        accel_peak = accel_features.peak_acceleration if accel_features else 0.0
        accel_rms = accel_features.rms_vibration if accel_features else 0.0
        
        # Check accelerometer thresholds
        accel_detected = (
            accel_peak >= self.config['accel_peak_threshold'] or
            accel_rms >= self.config['accel_rms_threshold']
        )
        
        # Get GPS from accel features if not provided
        if latitude is None and accel_features and accel_features.speed is not None:
            # Speed available suggests GPS data exists elsewhere
            pass
        
        # Perform fusion based on method
        if self.method == "rule_based":
            result = self._fuse_rule_based(
                vision_detected, vision_conf,
                accel_detected, accel_peak, accel_rms,
                vision_features, accel_features, accel_severity
            )
        elif self.method == "weighted_average":
            result = self._fuse_weighted(
                vision_detected, vision_conf,
                accel_detected, accel_peak, accel_rms,
                accel_severity
            )
        else:
            self.logger.warning(f"Unknown method {self.method}, using rule_based")
            result = self._fuse_rule_based(
                vision_detected, vision_conf,
                accel_detected, accel_peak, accel_rms,
                vision_features, accel_features, accel_severity
            )
        
        detected, confidence, severity = result
        
        return FusionResult(
            pothole_detected=detected,
            confidence=confidence,
            severity=severity,
            vision_detected=vision_detected,
            vision_confidence=vision_conf,
            accel_detected=accel_detected,
            accel_peak=accel_peak,
            accel_rms=accel_rms,
            latitude=latitude,
            longitude=longitude,
            timestamp=timestamp,
            fusion_method=self.method,
            vision_features=vision_features,
            accel_features=accel_features
        )
    
    def _fuse_rule_based(
        self,
        vision_detected: bool,
        vision_conf: float,
        accel_detected: bool,
        accel_peak: float,
        accel_rms: float,
        vision_features: Optional[VisionFeatures],
        accel_features: Optional[AccelFeatures],
        accel_severity: Optional[SeverityPrediction]
    ) -> Tuple[bool, float, str]:
        """
        Rule-based fusion logic.
        
        Returns:
            Tuple of (detected, confidence, severity)
        """
        # Detection logic: Either modality can trigger detection
        # but both increase confidence
        
        vision_score = vision_conf if vision_detected else 0.0
        accel_score = self._compute_accel_score(accel_peak, accel_rms)
        
        # Combined score
        combined = (
            self.vision_weight * vision_score +
            self.accel_weight * accel_score
        )
        
        # Detection decision
        if vision_detected and accel_detected:
            # Both agree - high confidence
            detected = True
            confidence = min(1.0, combined * 1.2)  # Boost for agreement
        elif vision_detected or accel_detected:
            # Single modality
            detected = combined >= self.detection_threshold
            confidence = combined
        else:
            detected = False
            confidence = combined
        
        # Severity determination
        if not detected:
            severity = 'none'
        else:
            severity = self._determine_severity(
                vision_features, accel_features, 
                accel_peak, accel_rms, accel_severity
            )
        
        return detected, confidence, severity
    
    def _fuse_weighted(
        self,
        vision_detected: bool,
        vision_conf: float,
        accel_detected: bool,
        accel_peak: float,
        accel_rms: float,
        accel_severity: Optional[SeverityPrediction]
    ) -> Tuple[bool, float, str]:
        """
        Simple weighted average fusion.
        
        Returns:
            Tuple of (detected, confidence, severity)
        """
        vision_score = vision_conf if vision_detected else 0.0
        accel_score = self._compute_accel_score(accel_peak, accel_rms)
        
        combined = (
            self.vision_weight * vision_score +
            self.accel_weight * accel_score
        )
        
        detected = combined >= self.detection_threshold
        
        if not detected:
            severity = 'none'
        elif accel_severity:
            severity = accel_severity.severity
        else:
            # Simple severity from accel peak
            if accel_peak >= 1.5:
                severity = 'high'
            elif accel_peak >= 0.5:
                severity = 'medium'
            else:
                severity = 'low'
        
        return detected, combined, severity
    
    def _compute_accel_score(
        self,
        peak: float,
        rms: float
    ) -> float:
        """
        Convert accelerometer metrics to a 0-1 score.
        
        Uses sigmoid-like mapping based on typical pothole ranges.
        """
        # Normalize peak (0.3-2.0g maps to 0-1)
        peak_norm = np.clip((peak - 0.3) / 1.7, 0, 1)
        
        # Normalize RMS (0.15-0.8g maps to 0-1)
        rms_norm = np.clip((rms - 0.15) / 0.65, 0, 1)
        
        # Combine with equal weights
        return 0.6 * peak_norm + 0.4 * rms_norm
    
    def _determine_severity(
        self,
        vision_features: Optional[VisionFeatures],
        accel_features: Optional[AccelFeatures],
        accel_peak: float,
        accel_rms: float,
        accel_severity: Optional[SeverityPrediction]
    ) -> str:
        """
        Determine final severity using all available information.
        """
        # Use ML severity if available
        if accel_severity and accel_severity.severity != 'none':
            base_severity = accel_severity.severity
        else:
            # Rule-based fallback
            if accel_peak >= 1.5 or accel_rms >= 0.8:
                base_severity = 'high'
            elif accel_peak >= 0.5 or accel_rms >= 0.2:
                base_severity = 'medium'
            else:
                base_severity = 'low'
        
        # Apply severity overrides
        overrides = self.config['severity_overrides']
        
        # High impact override
        if (accel_peak >= overrides['high_impact']['accel_peak'] or
            accel_rms >= overrides['high_impact']['accel_rms']):
            return 'high'
        
        # Large visual detection override
        if vision_features:
            if (vision_features.bbox_area_normalized >= overrides['large_visual']['bbox_area']
                and vision_features.confidence >= overrides['large_visual']['vision_conf']):
                if base_severity == 'low':
                    return 'medium'
        
        return base_severity
    
    def fuse_batch(
        self,
        vision_results: List[Optional[VisionFeatures]],
        accel_results: List[Optional[AccelFeatures]],
        severity_predictions: Optional[List[SeverityPrediction]] = None
    ) -> List[FusionResult]:
        """
        Fuse multiple pairs of results.
        
        Args:
            vision_results: List of vision features
            accel_results: List of accelerometer features  
            severity_predictions: Optional list of severity predictions
            
        Returns:
            List of FusionResult objects
        """
        n = max(len(vision_results), len(accel_results))
        
        # Pad shorter list with None
        vision_padded = vision_results + [None] * (n - len(vision_results))
        accel_padded = accel_results + [None] * (n - len(accel_results))
        
        if severity_predictions:
            sev_padded = severity_predictions + [None] * (n - len(severity_predictions))
        else:
            sev_padded = [None] * n
        
        results = []
        for vis, acc, sev in zip(vision_padded, accel_padded, sev_padded):
            result = self.fuse(vis, acc, accel_severity=sev)
            results.append(result)
        
        return results
