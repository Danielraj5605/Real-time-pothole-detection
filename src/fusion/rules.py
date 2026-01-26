"""
Rule-Based Fusion Module

Configurable rule-based fusion logic for pothole detection.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..vision.features import VisionFeatures
from ..accelerometer.features import AccelFeatures
from ..utils import get_logger


@dataclass
class FusionRule:
    """A single fusion rule with conditions and output."""
    name: str
    conditions: Dict[str, Any]
    severity: str
    confidence_boost: float = 0.0
    
    def evaluate(
        self,
        vision: Optional[VisionFeatures],
        accel: Optional[AccelFeatures]
    ) -> Tuple[bool, str, float]:
        """
        Evaluate rule against features.
        
        Returns:
            Tuple of (matched, severity, confidence_boost)
        """
        for condition, threshold in self.conditions.items():
            # Parse condition
            if condition.startswith('vision_'):
                if vision is None:
                    return False, '', 0.0
                field = condition[7:]  # Remove 'vision_' prefix
                value = getattr(vision, field, 0.0)
            elif condition.startswith('accel_'):
                if accel is None:
                    return False, '', 0.0
                field = condition[6:]  # Remove 'accel_' prefix
                value = getattr(accel, field, 0.0)
            else:
                continue
            
            # Evaluate threshold
            if isinstance(threshold, dict):
                if 'min' in threshold and value < threshold['min']:
                    return False, '', 0.0
                if 'max' in threshold and value > threshold['max']:
                    return False, '', 0.0
            else:
                if value < threshold:
                    return False, '', 0.0
        
        return True, self.severity, self.confidence_boost


class RuleBasedFusion:
    """
    Rule-based fusion system with configurable rules.
    
    Allows defining complex fusion logic through rules that can:
    - Check thresholds on vision and accelerometer features
    - Set severity based on conditions
    - Boost confidence for specific patterns
    
    Example:
        fusion = RuleBasedFusion()
        fusion.add_rule(FusionRule(
            name="high_impact",
            conditions={'accel_peak_acceleration': 2.0},
            severity='high',
            confidence_boost=0.2
        ))
        severity, conf = fusion.evaluate(vision_feats, accel_feats)
    """
    
    def __init__(self):
        """Initialize with default rules."""
        self.logger = get_logger("fusion.rules")
        self.rules: List[FusionRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default fusion rules."""
        # High severity rules
        self.add_rule(FusionRule(
            name="extreme_impact",
            conditions={
                'accel_peak_acceleration': 2.5
            },
            severity='high',
            confidence_boost=0.3
        ))
        
        self.add_rule(FusionRule(
            name="high_rms_vibration",
            conditions={
                'accel_rms_vibration': 1.0
            },
            severity='high',
            confidence_boost=0.2
        ))
        
        self.add_rule(FusionRule(
            name="large_visual_confident",
            conditions={
                'vision_bbox_area_normalized': 0.1,
                'vision_confidence': 0.85
            },
            severity='high',
            confidence_boost=0.15
        ))
        
        # Medium severity rules
        self.add_rule(FusionRule(
            name="moderate_impact",
            conditions={
                'accel_peak_acceleration': 1.0
            },
            severity='medium',
            confidence_boost=0.1
        ))
        
        self.add_rule(FusionRule(
            name="dual_modality_agreement",
            conditions={
                'vision_confidence': 0.6,
                'accel_peak_acceleration': 0.5
            },
            severity='medium',
            confidence_boost=0.2
        ))
        
        self.logger.info(f"Loaded {len(self.rules)} default fusion rules")
    
    def add_rule(self, rule: FusionRule):
        """Add a rule to the rule set."""
        self.rules.append(rule)
    
    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == name:
                del self.rules[i]
                return True
        return False
    
    def clear_rules(self):
        """Remove all rules."""
        self.rules.clear()
    
    def evaluate(
        self,
        vision: Optional[VisionFeatures],
        accel: Optional[AccelFeatures],
        base_confidence: float = 0.5
    ) -> Tuple[str, float]:
        """
        Evaluate all rules and determine final severity.
        
        Args:
            vision: Vision features
            accel: Accelerometer features
            base_confidence: Base confidence before rule evaluation
            
        Returns:
            Tuple of (severity, confidence)
        """
        # Track matched rules
        matched_rules = []
        
        for rule in self.rules:
            matched, severity, boost = rule.evaluate(vision, accel)
            if matched:
                matched_rules.append((rule, severity, boost))
        
        if not matched_rules:
            # No rules matched - use defaults
            return self._default_severity(vision, accel), base_confidence
        
        # Find highest severity matched
        severity_order = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        
        best_severity = 'low'
        total_boost = 0.0
        
        for rule, severity, boost in matched_rules:
            if severity_order.get(severity, 0) > severity_order.get(best_severity, 0):
                best_severity = severity
            total_boost += boost
        
        final_confidence = min(1.0, base_confidence + total_boost)
        
        self.logger.debug(
            f"Matched {len(matched_rules)} rules: "
            f"severity={best_severity}, conf={final_confidence:.2f}"
        )
        
        return best_severity, final_confidence
    
    def _default_severity(
        self,
        vision: Optional[VisionFeatures],
        accel: Optional[AccelFeatures]
    ) -> str:
        """Default severity when no rules match."""
        # Check if anything detected at all
        vision_detected = vision is not None and vision.detected
        accel_detected = accel is not None and accel.peak_acceleration > 0.3
        
        if not vision_detected and not accel_detected:
            return 'none'
        
        # Basic severity from accelerometer
        if accel:
            if accel.peak_acceleration >= 1.5:
                return 'high'
            elif accel.peak_acceleration >= 0.5:
                return 'medium'
        
        return 'low'
    
    def get_rule_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all rules."""
        return [
            {
                'name': rule.name,
                'conditions': rule.conditions,
                'severity': rule.severity,
                'confidence_boost': rule.confidence_boost
            }
            for rule in self.rules
        ]
