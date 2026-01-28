"""
Multimodal Fusion Service - Domain Logic
"""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FusionResult:
    """Result of multimodal fusion"""
    is_pothole_detected: bool
    confidence: float
    accel_peak: float
    bbox_area: int
    fusion_score: float
    
    
class FusionService:
    """
    Multimodal fusion logic for combining vision and accelerometer data.
    Pure domain logic - no infrastructure dependencies.
    """
    
    def __init__(
        self,
        vision_weight: float = 0.6,
        accel_weight: float = 0.4,
        fusion_threshold: float = 0.5
    ):
        """
        Initialize fusion service with weights.
        
        Args:
            vision_weight: Weight for vision detection (0-1)
            accel_weight: Weight for accelerometer data (0-1)
            fusion_threshold: Minimum fusion score to confirm detection
        """
        self.vision_weight = vision_weight
        self.accel_weight = accel_weight
        self.fusion_threshold = fusion_threshold
    
    def fuse(
        self,
        visual_detections: List[Dict[str, Any]],
        acceleration_data: Dict[str, float],
        min_confidence: float = 0.5,
        accel_threshold: float = 1.5
    ) -> FusionResult:
        """
        Fuse visual and accelerometer data to confirm pothole detection.
        
        Args:
            visual_detections: List of YOLO detections with confidence and bbox
            acceleration_data: Dict with x, y, z acceleration values
            min_confidence: Minimum vision confidence threshold
            accel_threshold: Minimum acceleration threshold (g-force)
            
        Returns:
            FusionResult with detection decision and metrics
        """
        # Extract best visual detection
        vision_confidence = 0.0
        bbox_area = 0
        
        if visual_detections:
            best_detection = max(visual_detections, key=lambda d: d.get('confidence', 0))
            vision_confidence = best_detection.get('confidence', 0)
            
            # Calculate bounding box area
            bbox = best_detection.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                bbox_area = int(width * height)
        
        # Calculate acceleration magnitude
        accel_x = acceleration_data.get('x', 0)
        accel_y = acceleration_data.get('y', 0)
        accel_z = acceleration_data.get('z', 0)
        accel_peak = (accel_x**2 + accel_y**2 + accel_z**2) ** 0.5
        
        # Normalize scores (0-1)
        vision_score = min(vision_confidence, 1.0)
        accel_score = min(accel_peak / 3.0, 1.0)  # Normalize assuming max 3g
        
        # Calculate weighted fusion score
        fusion_score = (
            self.vision_weight * vision_score +
            self.accel_weight * accel_score
        )
        
        # Decision logic
        is_detected = (
            fusion_score >= self.fusion_threshold and
            vision_confidence >= min_confidence and
            accel_peak >= accel_threshold
        )
        
        return FusionResult(
            is_pothole_detected=is_detected,
            confidence=vision_confidence,
            accel_peak=accel_peak,
            bbox_area=bbox_area,
            fusion_score=fusion_score
        )
