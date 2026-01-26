"""
Pothole Detector Module

YOLOv8-based inference pipeline for pothole detection with support for
both image files and numpy arrays (for future live camera integration).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Please install ultralytics: pip install ultralytics")

from ..utils import get_logger


@dataclass
class Detection:
    """Represents a single pothole detection."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get bounding box area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width / height)."""
        return self.width / max(self.height, 1e-6)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.center,
            'area': self.area,
            'width': self.width,
            'height': self.height,
            'aspect_ratio': self.aspect_ratio
        }


class PotholeDetector:
    """
    YOLOv8-based pothole detector.
    
    Provides inference capabilities with support for:
    - Single image inference
    - Batch inference
    - Live frame inference (numpy arrays)
    - Visualization with bounding boxes
    
    Example:
        detector = PotholeDetector("models/weights/pothole_best.pt")
        detections = detector.detect("image.jpg")
        for det in detections:
            print(f"Pothole: {det.confidence:.2f}")
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained YOLOv8 weights (.pt file)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.logger = get_logger("vision.detector")
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self._load_model()
        
        # Class mapping (will be populated from model)
        self.class_names: Dict[int, str] = {}
    
    def _load_model(self):
        """Load the YOLO model."""
        if not self.model_path.exists():
            self.logger.warning(
                f"Model not found at {self.model_path}. "
                "Using pretrained yolov8n.pt"
            )
            self.model = YOLO("yolov8n.pt")
        else:
            self.logger.info(f"Loading model from {self.model_path}")
            self.model = YOLO(str(self.model_path))
        
        # Move to device
        self.model.to(self.device)
        
        # Get class names from model
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
            self.logger.info(f"Classes: {self.class_names}")
    
    def detect(
        self,
        source: Union[str, Path, np.ndarray],
        augment: bool = False,
        verbose: bool = False
    ) -> List[Detection]:
        """
        Run detection on an image.
        
        Args:
            source: Image path, file, or numpy array (BGR format)
            augment: Enable test-time augmentation
            verbose: Print detailed output
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            source,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            augment=augment,
            verbose=verbose,
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
                
                detection = Detection(
                    bbox=(float(bbox[0]), float(bbox[1]), 
                          float(bbox[2]), float(bbox[3])),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                )
                detections.append(detection)
        
        self.logger.debug(f"Detected {len(detections)} pothole(s)")
        return detections
    
    def detect_batch(
        self,
        sources: List[Union[str, Path, np.ndarray]],
        batch_size: int = 8
    ) -> List[List[Detection]]:
        """
        Run detection on multiple images.
        
        Args:
            sources: List of image paths or numpy arrays
            batch_size: Batch size for inference
            
        Returns:
            List of detection lists (one per image)
        """
        all_detections = []
        
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i + batch_size]
            results = self.model(
                batch,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            for result in results:
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes
                    for j in range(len(boxes)):
                        bbox = boxes.xyxy[j].cpu().numpy()
                        conf = float(boxes.conf[j].cpu().numpy())
                        cls_id = int(boxes.cls[j].cpu().numpy())
                        cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                        
                        detections.append(Detection(
                            bbox=(float(bbox[0]), float(bbox[1]),
                                  float(bbox[2]), float(bbox[3])),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name
                        ))
                all_detections.append(detections)
        
        return all_detections
    
    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        detections: Optional[List[Detection]] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Draw detections on image.
        
        Args:
            image: Image path or numpy array
            detections: List of detections (if None, runs detection first)
            show_labels: Show class labels
            show_confidence: Show confidence scores
            color: Bounding box color (BGR)
            thickness: Line thickness
            font_scale: Font scale for labels
            
        Returns:
            Image with drawn detections (BGR numpy array)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        # Run detection if needed
        if detections is None:
            detections = self.detect(image)
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(det.class_name)
                if show_confidence:
                    label_parts.append(f"{det.confidence:.2f}")
                label = " ".join(label_parts)
                
                # Get text size
                (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                # Draw label background
                cv2.rectangle(
                    img, (x1, y1 - h - 10), (x1 + w, y1), color, -1
                )
                
                # Draw label text
                cv2.putText(
                    img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness
                )
        
        return img
    
    def get_best_detection(
        self,
        detections: List[Detection]
    ) -> Optional[Detection]:
        """
        Get the detection with highest confidence.
        
        Args:
            detections: List of detections
            
        Returns:
            Best detection or None if empty
        """
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)


class ImageDataSource:
    """
    Abstract data source for images.
    Designed to be replaced with live camera feed in future.
    """
    
    def __init__(self, source_path: Union[str, Path]):
        """
        Initialize with path to images.
        
        Args:
            source_path: Path to image file or directory
        """
        self.source_path = Path(source_path)
        self._images: List[Path] = []
        self._index = 0
        
        if self.source_path.is_dir():
            self._images = sorted(
                list(self.source_path.glob("*.jpg")) +
                list(self.source_path.glob("*.png")) +
                list(self.source_path.glob("*.jpeg"))
            )
        elif self.source_path.is_file():
            self._images = [self.source_path]
    
    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self) -> np.ndarray:
        if self._index >= len(self._images):
            raise StopIteration
        
        image_path = self._images[self._index]
        self._index += 1
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    def __len__(self) -> int:
        return len(self._images)
    
    def get_path(self, index: int) -> Optional[Path]:
        """Get path of image at index."""
        if 0 <= index < len(self._images):
            return self._images[index]
        return None
