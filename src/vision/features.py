"""
Vision Feature Extractor Module

Extracts normalized features from detections for multimodal fusion.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .detector import Detection
from ..utils import get_logger


@dataclass
class VisionFeatures:
    """Features extracted from vision detection for fusion."""
    
    # Detection present
    detected: bool
    
    # Core detection metrics
    confidence: float
    bbox_area: float
    bbox_area_normalized: float
    aspect_ratio: float
    
    # Additional metrics
    center_x_normalized: float
    center_y_normalized: float
    num_detections: int
    max_confidence: float
    avg_confidence: float
    total_area_normalized: float
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array for ML models."""
        return np.array([
            float(self.detected),
            self.confidence,
            self.bbox_area_normalized,
            self.aspect_ratio,
            self.center_x_normalized,
            self.center_y_normalized,
            self.num_detections,
            self.max_confidence,
            self.avg_confidence,
            self.total_area_normalized
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'detected': self.detected,
            'confidence': self.confidence,
            'bbox_area': self.bbox_area,
            'bbox_area_normalized': self.bbox_area_normalized,
            'aspect_ratio': self.aspect_ratio,
            'center_x_normalized': self.center_x_normalized,
            'center_y_normalized': self.center_y_normalized,
            'num_detections': self.num_detections,
            'max_confidence': self.max_confidence,
            'avg_confidence': self.avg_confidence,
            'total_area_normalized': self.total_area_normalized
        }
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get ordered list of feature names."""
        return [
            'detected',
            'confidence',
            'bbox_area_normalized',
            'aspect_ratio',
            'center_x_normalized',
            'center_y_normalized',
            'num_detections',
            'max_confidence',
            'avg_confidence',
            'total_area_normalized'
        ]


class VisionFeatureExtractor:
    """
    Extracts normalized features from vision detections.
    
    Features are designed for multimodal fusion with accelerometer data.
    All spatial features are normalized to [0, 1] range.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.logger = get_logger("vision.features")
    
    def extract(
        self,
        detections: List[Detection],
        image_width: int,
        image_height: int
    ) -> VisionFeatures:
        """
        Extract features from a list of detections.
        
        Args:
            detections: List of Detection objects
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            VisionFeatures dataclass
        """
        image_area = image_width * image_height
        
        if not detections:
            return VisionFeatures(
                detected=False,
                confidence=0.0,
                bbox_area=0.0,
                bbox_area_normalized=0.0,
                aspect_ratio=0.0,
                center_x_normalized=0.5,
                center_y_normalized=0.5,
                num_detections=0,
                max_confidence=0.0,
                avg_confidence=0.0,
                total_area_normalized=0.0
            )
        
        # Get best detection (highest confidence)
        best_det = max(detections, key=lambda d: d.confidence)
        
        # Calculate aggregate metrics
        confidences = [d.confidence for d in detections]
        areas = [d.area for d in detections]
        
        # Normalize center position
        center_x, center_y = best_det.center
        center_x_norm = center_x / image_width
        center_y_norm = center_y / image_height
        
        return VisionFeatures(
            detected=True,
            confidence=best_det.confidence,
            bbox_area=best_det.area,
            bbox_area_normalized=best_det.area / image_area,
            aspect_ratio=best_det.aspect_ratio,
            center_x_normalized=center_x_norm,
            center_y_normalized=center_y_norm,
            num_detections=len(detections),
            max_confidence=max(confidences),
            avg_confidence=np.mean(confidences),
            total_area_normalized=sum(areas) / image_area
        )
    
    def extract_from_result(
        self,
        result,
        image_shape: Tuple[int, int, int]
    ) -> VisionFeatures:
        """
        Extract features directly from YOLO result object.
        
        Args:
            result: YOLO inference result
            image_shape: (height, width, channels) tuple
            
        Returns:
            VisionFeatures dataclass
        """
        height, width = image_shape[:2]
        image_area = width * height
        
        if result.boxes is None or len(result.boxes) == 0:
            return VisionFeatures(
                detected=False,
                confidence=0.0,
                bbox_area=0.0,
                bbox_area_normalized=0.0,
                aspect_ratio=0.0,
                center_x_normalized=0.5,
                center_y_normalized=0.5,
                num_detections=0,
                max_confidence=0.0,
                avg_confidence=0.0,
                total_area_normalized=0.0
            )
        
        boxes = result.boxes
        
        # Extract all confidences and boxes
        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()
        
        # Find best detection
        best_idx = np.argmax(confs)
        best_conf = float(confs[best_idx])
        best_box = xyxys[best_idx]
        
        # Calculate box metrics
        x1, y1, x2, y2 = best_box
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        aspect_ratio = box_width / max(box_height, 1e-6)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate total area
        total_area = 0
        for box in xyxys:
            w = box[2] - box[0]
            h = box[3] - box[1]
            total_area += w * h
        
        return VisionFeatures(
            detected=True,
            confidence=best_conf,
            bbox_area=box_area,
            bbox_area_normalized=box_area / image_area,
            aspect_ratio=aspect_ratio,
            center_x_normalized=center_x / width,
            center_y_normalized=center_y / height,
            num_detections=len(boxes),
            max_confidence=float(np.max(confs)),
            avg_confidence=float(np.mean(confs)),
            total_area_normalized=total_area / image_area
        )
    
    def compute_severity_hint(
        self,
        features: VisionFeatures,
        area_threshold_high: float = 0.1,
        area_threshold_medium: float = 0.03,
        conf_threshold: float = 0.7
    ) -> str:
        """
        Compute preliminary severity based on visual features only.
        
        This is used as a hint for the fusion engine.
        
        Args:
            features: Extracted vision features
            area_threshold_high: Normalized area threshold for high severity
            area_threshold_medium: Normalized area threshold for medium severity
            conf_threshold: Confidence threshold for severity upgrade
            
        Returns:
            Severity string: 'low', 'medium', 'high', or 'none'
        """
        if not features.detected:
            return 'none'
        
        # Large visual area + high confidence = high severity
        if (features.bbox_area_normalized > area_threshold_high and 
            features.confidence > conf_threshold):
            return 'high'
        
        # Medium area or multiple detections
        if (features.bbox_area_normalized > area_threshold_medium or 
            features.num_detections > 2):
            return 'medium'
        
        return 'low'
