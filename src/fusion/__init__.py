"""
Multimodal Fusion Module

Combines vision and accelerometer features for robust pothole detection.
"""

from .engine import FusionEngine, FusionResult
from .rules import RuleBasedFusion
from .alerts import AlertManager, Alert

__all__ = [
    'FusionEngine',
    'FusionResult', 
    'RuleBasedFusion',
    'AlertManager',
    'Alert'
]
