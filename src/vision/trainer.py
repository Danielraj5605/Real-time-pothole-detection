"""
Vision Trainer Module

Training pipeline for YOLOv8 pothole detection model.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")

from ..utils import get_logger, get_config


class VisionTrainer:
    """
    YOLOv8 training pipeline for pothole detection.
    
    Handles:
    - Model initialization and configuration
    - Training with configurable hyperparameters
    - Validation and metrics tracking
    - Model export and weight saving
    
    Example:
        trainer = VisionTrainer()
        trainer.train("datasets/pothole_dataset.yaml")
        trainer.export("models/weights/pothole_best.pt")
    """
    
    def __init__(
        self,
        model_type: str = "yolov8n",
        pretrained_weights: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_type: YOLO model type (yolov8n, yolov8s, yolov8m, etc.)
            pretrained_weights: Path to pretrained weights or None for default
        """
        self.logger = get_logger("vision.trainer")
        self.model_type = model_type
        
        # Load configuration
        try:
            self.config = get_config()
            self.vision_config = self.config.get_vision_config()
        except Exception:
            self.logger.warning("Config not found, using defaults")
            self.vision_config = {}
        
        # Initialize model
        if pretrained_weights and Path(pretrained_weights).exists():
            self.model = YOLO(pretrained_weights)
            self.logger.info(f"Loaded pretrained weights: {pretrained_weights}")
        else:
            model_file = f"{model_type}.pt"
            self.model = YOLO(model_file)
            self.logger.info(f"Initialized {model_type} with COCO weights")
        
        self.training_results = None
        self.best_weights_path: Optional[Path] = None
    
    def train(
        self,
        data_yaml: str,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        image_size: int = 640,
        patience: int = 20,
        project: str = "models/yolo_training",
        name: Optional[str] = None,
        resume: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the YOLO model.
        
        Args:
            data_yaml: Path to dataset YAML configuration
            epochs: Number of training epochs
            batch_size: Training batch size
            image_size: Input image size
            patience: Early stopping patience
            project: Output project directory
            name: Run name (auto-generated if None)
            resume: Resume from last checkpoint
            **kwargs: Additional training arguments
            
        Returns:
            Training results dictionary
        """
        # Get training config
        train_config = self.vision_config.get('training', {})
        
        # Set defaults from config
        epochs = epochs or train_config.get('epochs', 100)
        batch_size = batch_size or train_config.get('batch_size', 16)
        
        # Generate run name
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"pothole_{self.model_type}_{timestamp}"
        
        self.logger.info(f"Starting training: {epochs} epochs, batch={batch_size}")
        self.logger.info(f"Dataset: {data_yaml}")
        
        # Create output directory
        output_dir = Path(project)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': image_size,
            'patience': patience,
            'project': str(project),
            'name': name,
            'exist_ok': True,
            'pretrained': True,
            'resume': resume,
            'verbose': True,
            'save': True,
            'save_period': 10,
            'val': True,
            'plots': True,
            # Augmentation
            'augment': train_config.get('augment', True),
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
        }
        
        # Add optimizer settings from config
        if 'lr0' in train_config:
            train_args['lr0'] = train_config['lr0']
        if 'lrf' in train_config:
            train_args['lrf'] = train_config['lrf']
        if 'momentum' in train_config:
            train_args['momentum'] = train_config['momentum']
        if 'weight_decay' in train_config:
            train_args['weight_decay'] = train_config['weight_decay']
        
        # Override with kwargs
        train_args.update(kwargs)
        
        # Run training
        try:
            results = self.model.train(**train_args)
            self.training_results = results
            
            # Find best weights
            run_dir = Path(project) / name
            best_weights = run_dir / "weights" / "best.pt"
            
            if best_weights.exists():
                self.best_weights_path = best_weights
                self.logger.info(f"Best weights saved: {best_weights}")
            
            return self._extract_metrics(results)
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _extract_metrics(self, results) -> Dict[str, Any]:
        """Extract training metrics from results."""
        metrics = {
            'training_complete': True,
        }
        
        if hasattr(results, 'results_dict'):
            metrics.update(results.results_dict)
        
        return metrics
    
    def validate(
        self,
        data_yaml: Optional[str] = None,
        weights: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run validation on the model.
        
        Args:
            data_yaml: Dataset YAML (uses training data if None)
            weights: Model weights (uses best from training if None)
            **kwargs: Additional validation arguments
            
        Returns:
            Validation metrics dictionary
        """
        if weights:
            model = YOLO(weights)
        elif self.best_weights_path:
            model = YOLO(str(self.best_weights_path))
        else:
            model = self.model
        
        val_args = {
            'data': data_yaml,
            'verbose': True,
            'plots': True
        }
        val_args.update(kwargs)
        
        self.logger.info("Running validation...")
        results = model.val(**val_args)
        
        return self._extract_val_metrics(results)
    
    def _extract_val_metrics(self, results) -> Dict[str, Any]:
        """Extract validation metrics."""
        metrics = {}
        
        if hasattr(results, 'box'):
            box = results.box
            if hasattr(box, 'map'):
                metrics['mAP50-95'] = float(box.map)
            if hasattr(box, 'map50'):
                metrics['mAP50'] = float(box.map50)
            if hasattr(box, 'map75'):
                metrics['mAP75'] = float(box.map75)
        
        return metrics
    
    def export(
        self,
        output_path: str,
        format: str = "pt"
    ) -> Path:
        """
        Export the best model weights to specified location.
        
        Args:
            output_path: Destination path for weights
            format: Export format (pt, onnx, etc.)
            
        Returns:
            Path to exported weights
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.best_weights_path and self.best_weights_path.exists():
            if format == "pt":
                shutil.copy(self.best_weights_path, output_path)
                self.logger.info(f"Exported weights to: {output_path}")
                return output_path
            else:
                # Export to other formats
                model = YOLO(str(self.best_weights_path))
                exported = model.export(format=format)
                
                if exported:
                    shutil.move(exported, output_path)
                    return output_path
        
        raise FileNotFoundError("No trained weights available to export")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        info = {
            'model_type': self.model_type,
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
            'layers': len(list(self.model.model.modules())),
        }
        
        if self.best_weights_path:
            info['best_weights'] = str(self.best_weights_path)
        
        return info
