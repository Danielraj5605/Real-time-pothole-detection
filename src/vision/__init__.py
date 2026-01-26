"""
Vision Pipeline Module

YOLOv8-based pothole detection with feature extraction for multimodal fusion.
"""

from .detector import PotholeDetector
from .trainer import VisionTrainer
from .features import VisionFeatureExtractor

__all__ = ['PotholeDetector', 'VisionTrainer', 'VisionFeatureExtractor']
