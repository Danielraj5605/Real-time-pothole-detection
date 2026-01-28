"""
YOLOv8 Detector Adapter
Wraps existing YOLOv8 detection to implement DetectorInterface
"""
import logging
from pathlib import Path
from typing import List
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from ..interfaces.detector_interface import DetectorInterface, DetectionResult


class YOLOv8Detector(DetectorInterface):
    """
    YOLOv8 detector adapter.
    Wraps the existing vision.detector module to implement DetectorInterface.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = None
    ):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to trained model weights
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.class_names = {}
    
    def initialize(self) -> bool:
        """Initialize the YOLO model"""
        if YOLO is None:
            self.logger.error("ultralytics not installed. Install with: pip install ultralytics")
            return False
        
        try:
            if not self.model_path.exists():
                self.logger.warning(
                    f"Model not found at {self.model_path}. Using pretrained yolov8n.pt"
                )
                self.model = YOLO("yolov8n.pt")
            else:
                self.logger.info(f"Loading model from {self.model_path}")
                self.model = YOLO(str(self.model_path))
            
            # Move to device
            self.model.to(self.device)
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                self.logger.info(f"Model loaded with classes: {self.class_names}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run detection on an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detection results
        """
        if self.model is None:
            self.logger.error("Model not initialized")
            return []
        
        try:
            # Run inference
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
            
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    
                    detection = DetectionResult(
                        bbox=(float(bbox[0]), float(bbox[1]),
                              float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name
                    )
                    detections.append(detection)
            
            return detections
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def cleanup(self) -> None:
        """Release model resources"""
        if self.model is not None:
            del self.model
            self.model = None
            self.logger.info("Model released")
