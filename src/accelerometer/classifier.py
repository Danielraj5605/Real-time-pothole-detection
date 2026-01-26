"""
Severity Classifier Module

ML-based pothole severity classification from accelerometer features.
Supports Random Forest, Gradient Boosting, and synthetic data training.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from .features import AccelFeatures
from ..utils import get_logger


@dataclass
class SeverityPrediction:
    """Prediction result from severity classifier."""
    severity: str  # 'none', 'low', 'medium', 'high'
    confidence: float
    probabilities: Dict[str, float]


class SeverityClassifier:
    """
    ML classifier for pothole severity from accelerometer features.
    
    Supports:
    - Random Forest (default)
    - Gradient Boosting
    - Model persistence (save/load)
    - Synthetic training data generation
    
    Example:
        classifier = SeverityClassifier()
        classifier.train_synthetic()  # or train with real data
        prediction = classifier.predict(accel_features)
        print(f"Severity: {prediction.severity}")
    """
    
    SEVERITY_LEVELS = ['none', 'low', 'medium', 'high']
    
    def __init__(
        self,
        model_type: str = "random_forest",
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None
    ):
        """
        Initialize the classifier.
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
            model_path: Path to load pre-trained model
            scaler_path: Path to load pre-trained scaler
        """
        self.logger = get_logger("accel.classifier")
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Try to load existing model
        if model_path and Path(model_path).exists():
            self.load(model_path, scaler_path)
    
    def _create_model(self, **kwargs) -> Any:
        """Create a new model instance."""
        if self.model_type == "random_forest":
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(kwargs)
            return RandomForestClassifier(**default_params)
        
        elif self.model_type == "gradient_boosting":
            default_params = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            }
            default_params.update(kwargs)
            return GradientBoostingClassifier(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def generate_synthetic_data(
        self,
        n_samples_per_class: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data based on physical intuition.
        
        Args:
            n_samples_per_class: Number of samples per severity class
            
        Returns:
            Tuple of (X, y) arrays
        """
        np.random.seed(42)
        
        X_all = []
        y_all = []
        
        # Feature order: peak, peak_x, peak_y, peak_z, rms, rms_x, rms_y, rms_z,
        #                mean, std, p2p, crest, zcr, speed
        
        # Class 0: None (normal road)
        for _ in range(n_samples_per_class):
            peak = np.random.uniform(0.05, 0.25)
            rms = np.random.uniform(0.02, 0.1)
            features = [
                peak,  # peak_acceleration
                peak * np.random.uniform(0.2, 0.5),  # peak_x
                peak * np.random.uniform(0.2, 0.5),  # peak_y  
                peak * np.random.uniform(0.8, 1.0),  # peak_z
                rms,  # rms_vibration
                rms * np.random.uniform(0.3, 0.5),  # rms_x
                rms * np.random.uniform(0.3, 0.5),  # rms_y
                rms * np.random.uniform(0.6, 0.8),  # rms_z
                rms * np.random.uniform(0.8, 1.2),  # mean
                rms * np.random.uniform(0.5, 1.0),  # std
                peak * np.random.uniform(1.5, 2.0),  # p2p
                peak / max(rms, 0.01),  # crest_factor
                np.random.uniform(0.1, 0.3),  # zcr
                np.random.uniform(5, 40)  # speed
            ]
            X_all.append(features)
            y_all.append(0)
        
        # Class 1: Low severity (minor bumps)
        for _ in range(n_samples_per_class):
            peak = np.random.uniform(0.25, 0.6)
            rms = np.random.uniform(0.1, 0.2)
            features = [
                peak,
                peak * np.random.uniform(0.3, 0.6),
                peak * np.random.uniform(0.3, 0.6),
                peak * np.random.uniform(0.7, 1.0),
                rms,
                rms * np.random.uniform(0.3, 0.5),
                rms * np.random.uniform(0.3, 0.5),
                rms * np.random.uniform(0.6, 0.8),
                rms * np.random.uniform(0.8, 1.2),
                rms * np.random.uniform(0.8, 1.5),
                peak * np.random.uniform(1.5, 2.5),
                peak / max(rms, 0.01),
                np.random.uniform(0.15, 0.4),
                np.random.uniform(5, 40)
            ]
            X_all.append(features)
            y_all.append(1)
        
        # Class 2: Medium severity
        for _ in range(n_samples_per_class):
            peak = np.random.uniform(0.6, 1.5)
            rms = np.random.uniform(0.2, 0.5)
            features = [
                peak,
                peak * np.random.uniform(0.3, 0.7),
                peak * np.random.uniform(0.3, 0.7),
                peak * np.random.uniform(0.6, 1.0),
                rms,
                rms * np.random.uniform(0.3, 0.5),
                rms * np.random.uniform(0.3, 0.5),
                rms * np.random.uniform(0.5, 0.8),
                rms * np.random.uniform(0.6, 1.0),
                rms * np.random.uniform(1.0, 2.0),
                peak * np.random.uniform(1.8, 3.0),
                peak / max(rms, 0.01),
                np.random.uniform(0.2, 0.5),
                np.random.uniform(5, 35)
            ]
            X_all.append(features)
            y_all.append(2)
        
        # Class 3: High severity
        for _ in range(n_samples_per_class):
            peak = np.random.uniform(1.5, 4.0)
            rms = np.random.uniform(0.5, 1.5)
            features = [
                peak,
                peak * np.random.uniform(0.4, 0.8),
                peak * np.random.uniform(0.4, 0.8),
                peak * np.random.uniform(0.5, 1.0),
                rms,
                rms * np.random.uniform(0.3, 0.6),
                rms * np.random.uniform(0.3, 0.6),
                rms * np.random.uniform(0.5, 0.8),
                rms * np.random.uniform(0.5, 0.9),
                rms * np.random.uniform(1.5, 3.0),
                peak * np.random.uniform(2.0, 4.0),
                peak / max(rms, 0.01),
                np.random.uniform(0.25, 0.6),
                np.random.uniform(5, 30)
            ]
            X_all.append(features)
            y_all.append(3)
        
        X = np.array(X_all)
        y = np.array(y_all)
        
        # Shuffle
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices]
        
        self.logger.info(
            f"Generated {len(y)} synthetic samples "
            f"({n_samples_per_class} per class)"
        )
        
        return X, y
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            test_size: Fraction for test set
            **model_kwargs: Additional model parameters
            
        Returns:
            Training metrics dictionary
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        self.model = self._create_model(**model_kwargs)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='accuracy'
        )
        
        metrics = {
            'train_accuracy': self.model.score(X_train_scaled, y_train),
            'test_accuracy': self.model.score(X_test_scaled, y_test),
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'classification_report': classification_report(
                y_test, y_pred, target_names=self.SEVERITY_LEVELS
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        self.logger.info(
            f"Training complete: test_acc={metrics['test_accuracy']:.3f}, "
            f"cv={metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}"
        )
        
        return metrics
    
    def train_synthetic(self, n_samples_per_class: int = 500) -> Dict[str, Any]:
        """
        Train on synthetic data.
        
        Args:
            n_samples_per_class: Samples per severity class
            
        Returns:
            Training metrics
        """
        X, y = self.generate_synthetic_data(n_samples_per_class)
        return self.train(X, y)
    
    def predict(self, features: AccelFeatures) -> SeverityPrediction:
        """
        Predict severity from accelerometer features.
        
        Args:
            features: AccelFeatures object
            
        Returns:
            SeverityPrediction object
        """
        if not self.is_trained:
            self.logger.warning("Model not trained, using rule-based fallback")
            from .features import AccelFeatureExtractor
            extractor = AccelFeatureExtractor()
            severity, conf = extractor.compute_severity_rule_based(features)
            return SeverityPrediction(
                severity=severity,
                confidence=conf,
                probabilities={sev: 0.0 for sev in self.SEVERITY_LEVELS}
            )
        
        # Convert features to array and scale
        X = features.to_array().reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        pred_class = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        severity = self.SEVERITY_LEVELS[pred_class]
        confidence = float(probabilities[pred_class])
        
        prob_dict = {
            self.SEVERITY_LEVELS[i]: float(probabilities[i])
            for i in range(len(self.SEVERITY_LEVELS))
        }
        
        return SeverityPrediction(
            severity=severity,
            confidence=confidence,
            probabilities=prob_dict
        )
    
    def predict_batch(
        self,
        features_list: List[AccelFeatures]
    ) -> List[SeverityPrediction]:
        """
        Predict severity for multiple feature sets.
        
        Args:
            features_list: List of AccelFeatures objects
            
        Returns:
            List of SeverityPrediction objects
        """
        return [self.predict(f) for f in features_list]
    
    def save(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Save model and scaler to disk.
        
        Args:
            model_path: Path for model file
            scaler_path: Path for scaler file (defaults to model_path + '_scaler')
        """
        if not self.is_trained:
            raise ValueError("Model not trained, cannot save")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        if scaler_path is None:
            scaler_path = str(model_path).replace('.pkl', '_scaler.pkl')
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        self.logger.info(f"Saved model to {model_path}")
    
    def load(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Load model and scaler from disk.
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        if scaler_path is None:
            scaler_path = str(model_path).replace('.pkl', '_scaler.pkl')
        
        if Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        self.is_trained = True
        self.logger.info(f"Loaded model from {model_path}")
