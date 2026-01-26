"""
Accelerometer Pipeline Module

Signal processing and severity classification from accelerometer data.
"""

from .processor import AccelerometerProcessor
from .features import AccelFeatureExtractor, AccelFeatures
from .classifier import SeverityClassifier

__all__ = [
    'AccelerometerProcessor', 
    'AccelFeatureExtractor', 
    'AccelFeatures',
    'SeverityClassifier'
]
