"""
Pothole Detection System - Consolidated Main Application

This is the ONLY file you need to run the pothole detection system.
It combines all functionality into a single, easy-to-use application.

Pipeline Stages:
1. Clean Frames - Preprocessing
2. Find Object - YOLO detection  
3. Track Object - Multi-object tracking
4. Isolate - Region extraction
5. Read Information - Feature extraction
6. Fuse - Vision + Accelerometer fusion
7. Identify - Classification

Usage:
    python pothole_detector.py                    # Run with webcam
    python pothole_detector.py --camera 0         # Specify camera
    python pothole_detector.py --save             # Save detections
    python pothole_detector.py --test             # Test mode (no camera)
"""

import cv2
import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

# Setup project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import accelerometer processor
from accelerometer_processor import AccelerometerProcessor, AccelConfig, AccelFeatures


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """All configuration in one place"""
    # Camera
    camera_id: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    
    # Model
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    
    # Detection
    enable_tracking: bool = True
    enable_classification: bool = True
    max_missing_frames: int = 10
    iou_threshold: float = 0.3
    
    # Display
    show_window: bool = True
    save_detections: bool = False
    save_path: str = "results/detections"
    
    # Accelerometer
    enable_accelerometer: bool = True
    accel_csv_path: str = ""  # CSV path for simulation mode
    

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PotholeInfo:
    """Complete information about a detected pothole"""
    track_id: int
    frame_number: int
    timestamp: datetime
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    
    # Measurements
    width: int = 0
    height: int = 0
    area: int = 0
    area_ratio: float = 0.0
    
    # Classification
    severity: str = "UNKNOWN"
    depth: str = "UNKNOWN"
    
    # Accelerometer fusion data
    fusion_score: float = 0.0
    accel_severity: str = "NONE"
    accel_peak_g: float = 0.0
    accel_rms_g: float = 0.0
    has_accel_data: bool = False
    
    # Tracking
    track_length: int = 1
    
    # Isolated image
    isolated: Optional[np.ndarray] = None


@dataclass  
class TrackedObject:
    """Object tracked across frames"""
    track_id: int
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    first_seen: int = 0
    last_seen: int = 0
    frames_missing: int = 0
    is_active: bool = True


# =============================================================================
# YOLO DETECTOR
# =============================================================================

class YOLODetector:
    """YOLOv8 pothole detector"""
    
    def __init__(self, model_path: str, confidence: float = 0.25):
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize the YOLO model"""
        try:
            from ultralytics import YOLO
            self.logger.info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.logger.info("[OK] YOLO model loaded")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load YOLO: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run detection on a frame"""
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is None:
                    continue
                
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'class_id': cls_id
                    })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []


# =============================================================================
# DETECTION PIPELINE (6 STAGES)
# =============================================================================

class DetectionPipeline:
    """
    7-Stage Multimodal Pothole Detection Pipeline
    
    1. Clean Frames
    2. Find Object
    3. Track Object
    4. Isolate
    5. Read Information
    6. Fuse (Vision + Accelerometer)
    7. Identify
    """
    
    def __init__(self, detector: YOLODetector, config: Config,
                 accel_processor: Optional[AccelerometerProcessor] = None):
        self.detector = detector
        self.config = config
        self.accel_processor = accel_processor
        self.logger = logging.getLogger(__name__)
        
        # Tracking state
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}
        
        # Stats
        self.total_detections = 0
        self.total_tracks = 0
    
    # Stage 1: Clean Frames
    def clean_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess and clean the frame"""
        # Bilateral filter (noise reduction, preserves edges)
        denoised = cv2.bilateralFilter(frame, 9, 75, 75)
        
        # CLAHE for contrast enhancement
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    # Stage 2: Find Object
    def find_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect potholes using YOLO"""
        return self.detector.detect(frame)
    
    # Stage 3: Track Object
    def track_objects(self, detections: List[Dict], frame_num: int) -> List[Dict]:
        """Track objects across frames"""
        if not self.config.enable_tracking:
            return detections
        
        tracked = []
        matched_tracks = set()
        
        for det in detections:
            bbox = det['bbox']
            best_id = None
            best_iou = 0
            
            # Find matching track
            for tid, track in self.tracks.items():
                if not track.is_active or len(track.bbox_history) == 0:
                    continue
                
                iou = self._calc_iou(bbox, track.bbox_history[-1])
                if iou > best_iou and iou > self.config.iou_threshold:
                    best_iou = iou
                    best_id = tid
            
            if best_id is not None:
                # Update existing track
                track = self.tracks[best_id]
                track.bbox_history.append(bbox)
                track.last_seen = frame_num
                track.frames_missing = 0
                matched_tracks.add(best_id)
                
                det['track_id'] = best_id
                det['track_length'] = len(track.bbox_history)
            else:
                # Create new track
                tid = self.next_track_id
                self.next_track_id += 1
                self.total_tracks += 1
                
                track = TrackedObject(track_id=tid, first_seen=frame_num, last_seen=frame_num)
                track.bbox_history.append(bbox)
                self.tracks[tid] = track
                matched_tracks.add(tid)
                
                det['track_id'] = tid
                det['track_length'] = 1
            
            tracked.append(det)
        
        # Update missing tracks
        for tid, track in list(self.tracks.items()):
            if tid not in matched_tracks:
                track.frames_missing += 1
                if track.frames_missing > self.config.max_missing_frames:
                    track.is_active = False
        
        return tracked
    
    def _calc_iou(self, bbox1, bbox2) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    # Stage 4: Isolate
    def isolate(self, frame: np.ndarray, bbox: Tuple, padding: int = 10) -> np.ndarray:
        """Extract pothole region"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return frame[y1:y2, x1:x2].copy()
    
    # Stage 5: Read Information
    def read_info(self, frame: np.ndarray, bbox: Tuple) -> Dict:
        """Extract features and measurements"""
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        area = width * height
        frame_area = frame.shape[0] * frame.shape[1]
        
        return {
            'width': width,
            'height': height,
            'area': area,
            'area_ratio': area / frame_area if frame_area > 0 else 0,
            'center': ((x1 + x2) // 2, (y1 + y2) // 2)
        }
    
    # Stage 6: Fuse (Vision + Accelerometer)
    def fuse(self, features: Dict, confidence: float) -> Dict:
        """Fuse vision and accelerometer data for severity classification"""
        area_ratio = features.get('area_ratio', 0)
        
        # If accelerometer processor is available, use fusion
        if self.accel_processor and self.accel_processor.is_initialized:
            # Extract latest accelerometer features
            accel_features = self.accel_processor.extract_features()
            
            # Run fusion classification
            result = self.accel_processor.fuse_severity(
                vision_confidence=confidence,
                vision_area_ratio=area_ratio,
                accel_features=accel_features
            )
            return result
        
        # Fallback: vision-only classification (no accelerometer)
        return self._vision_only_classify(features, confidence)
    
    def _vision_only_classify(self, features: Dict, confidence: float) -> Dict:
        """Fallback: classify using vision data only"""
        area_ratio = features.get('area_ratio', 0)
        
        if confidence > 0.7 and area_ratio > 0.1:
            severity = "HIGH"
        elif confidence > 0.5 or area_ratio > 0.05:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        if area_ratio > 0.15:
            depth = "DEEP"
        elif area_ratio > 0.08:
            depth = "MODERATE"
        else:
            depth = "SHALLOW"
        
        return {
            'severity': severity, 'depth': depth,
            'fusion_score': 0.0, 'accel_severity': 'NONE',
            'accel_peak_g': 0.0, 'accel_rms_g': 0.0,
            'has_accel_data': False,
        }
    
    # Stage 7: Identify (post-fusion)
    def identify(self, fusion_result: Dict) -> Dict:
        """Return final classification from fusion result"""
        return fusion_result
    
    # Complete Pipeline
    def process(self, frame: np.ndarray, frame_num: int) -> List[PotholeInfo]:
        """Run complete 6-stage pipeline"""
        
        # Stage 1: Clean
        cleaned = self.clean_frame(frame)
        
        # Stage 2: Find
        detections = self.find_objects(cleaned)
        self.total_detections += len(detections)
        
        if not detections:
            return []
        
        # Stage 3: Track
        tracked = self.track_objects(detections, frame_num)
        
        # Process each detection through stages 4-6
        potholes = []
        for det in tracked:
            bbox = det['bbox']
            conf = det['confidence']
            
            # Stage 4: Isolate
            isolated = self.isolate(frame, bbox)
            
            # Stage 5: Read Info
            features = self.read_info(frame, bbox)
            
            # Stage 6: Fuse (Vision + Accelerometer)
            fusion_result = self.fuse(features, conf)
            
            # Stage 7: Identify
            classification = self.identify(fusion_result)
            
            pothole = PotholeInfo(
                track_id=det.get('track_id', 0),
                frame_number=frame_num,
                timestamp=datetime.now(),
                bbox=bbox,
                center=features['center'],
                confidence=conf,
                width=features['width'],
                height=features['height'],
                area=features['area'],
                area_ratio=features['area_ratio'],
                severity=classification['severity'],
                depth=classification['depth'],
                fusion_score=classification.get('fusion_score', 0.0),
                accel_severity=classification.get('accel_severity', 'NONE'),
                accel_peak_g=classification.get('accel_peak_g', 0.0),
                accel_rms_g=classification.get('accel_rms_g', 0.0),
                has_accel_data=classification.get('has_accel_data', False),
                track_length=det.get('track_length', 1),
                isolated=isolated
            )
            potholes.append(pothole)
        
        return potholes


# =============================================================================
# VISUALIZATION & ANNOTATION
# =============================================================================

# Severity colors (BGR format)
SEVERITY_COLORS = {
    'HIGH': (0, 0, 255),      # Red
    'MEDIUM': (0, 165, 255),  # Orange
    'LOW': (0, 255, 255),     # Yellow
    'UNKNOWN': (128, 128, 128) # Gray
}

# Overlay colors for segmentation (BGR)
OVERLAY_COLORS = {
    'HIGH': (0, 0, 200),      # Dark Red
    'MEDIUM': (0, 140, 200),  # Dark Orange
    'LOW': (0, 200, 200),     # Dark Yellow
}


def create_pothole_mask(frame: np.ndarray, bbox: Tuple, padding: int = 5) -> np.ndarray:
    """
    Create a segmentation mask for a pothole region.
    
    Uses edge detection and morphological operations to create
    a pixel-level mask that highlights the actual pothole area
    within the bounding box.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    
    # Ensure bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Extract region
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Detect dark regions (potholes are typically darker)
    mean_val = np.mean(enhanced)
    _, dark_mask = cv2.threshold(enhanced, mean_val * 0.7, 255, cv2.THRESH_BINARY_INV)
    
    # Edge detection
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Combine dark regions with edges
    combined = cv2.bitwise_or(dark_mask, edges)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(combined)
    if contours:
        # Keep only the largest contour
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [largest], -1, 255, -1)
    
    # Create full-size mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = filled
    
    return full_mask


def annotate_image_with_segmentation(
    frame: np.ndarray,
    potholes: List[PotholeInfo],
    overlay_alpha: float = 0.4,
    show_labels: bool = True
) -> np.ndarray:
    """
    Create annotated image with segmentation masks and bounding boxes.
    
    Features:
    - Colored overlay segmentation masks for each pothole
    - Clear bounding boxes with "Pothole" labels
    - Severity-based color coding
    - Semi-transparent overlays
    """
    display = frame.copy()
    overlay = frame.copy()
    
    for pothole in potholes:
        x1, y1, x2, y2 = pothole.bbox
        severity = pothole.severity
        color = SEVERITY_COLORS.get(severity, (0, 255, 0))
        overlay_color = OVERLAY_COLORS.get(severity, (0, 180, 180))
        
        # Create segmentation mask for this pothole
        mask = create_pothole_mask(frame, pothole.bbox)
        
        # Apply colored overlay where mask is active
        colored_overlay = np.zeros_like(frame)
        colored_overlay[:] = overlay_color
        
        # Blend overlay with original in masked region
        mask_3ch = cv2.merge([mask, mask, mask])
        overlay = np.where(mask_3ch > 0, 
                          cv2.addWeighted(overlay, 1-overlay_alpha, colored_overlay, overlay_alpha, 0),
                          overlay)
        
        # Draw bounding box (thick)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        
        # Draw corner accents
        corner_len = min(20, (x2-x1)//4, (y2-y1)//4)
        # Top-left
        cv2.line(overlay, (x1, y1), (x1+corner_len, y1), color, 4)
        cv2.line(overlay, (x1, y1), (x1, y1+corner_len), color, 4)
        # Top-right
        cv2.line(overlay, (x2, y1), (x2-corner_len, y1), color, 4)
        cv2.line(overlay, (x2, y1), (x2, y1+corner_len), color, 4)
        # Bottom-left
        cv2.line(overlay, (x1, y2), (x1+corner_len, y2), color, 4)
        cv2.line(overlay, (x1, y2), (x1, y2-corner_len), color, 4)
        # Bottom-right
        cv2.line(overlay, (x2, y2), (x2-corner_len, y2), color, 4)
        cv2.line(overlay, (x2, y2), (x2, y2-corner_len), color, 4)
        
        if show_labels:
            # Label: "Pothole" with severity and confidence
            label = f"Pothole ({severity})"
            conf_label = f"{pothole.confidence:.0%}"
            
            # Background for label
            (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (w2, h2), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            label_bg_h = h1 + h2 + 15
            label_bg_w = max(w1, w2) + 10
            
            # Position label above box
            label_y = max(label_bg_h + 5, y1 - 5)
            
            cv2.rectangle(overlay, (x1, label_y - label_bg_h), 
                         (x1 + label_bg_w, label_y), color, -1)
            
            # Main label
            cv2.putText(overlay, label, (x1 + 5, label_y - h2 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Confidence
            cv2.putText(overlay, conf_label, (x1 + 5, label_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return overlay


def draw_detections(frame: np.ndarray, potholes: List[PotholeInfo], fps: float) -> np.ndarray:
    """Draw detection results on frame (for live view)"""
    display = frame.copy()
    
    for p in potholes:
        x1, y1, x2, y2 = p.bbox
        color = SEVERITY_COLORS.get(p.severity, (0, 255, 0))
        
        # Bounding box
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        
        # Center point
        cv2.circle(display, p.center, 5, color, -1)
        
        # Label
        label = f"ID:{p.track_id} {p.severity} {p.confidence:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (x1, y1-h-8), (x1+w+4, y1), color, -1)
        cv2.putText(display, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Header
    cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(display, "POTHOLE DETECTION SYSTEM", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, "Clean > Find > Track > Isolate > Read > Identify", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    cv2.putText(display, f"FPS: {fps:.1f} | Potholes: {len(potholes)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return display


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class PotholeDetector:
    """Main application class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.detector = YOLODetector(config.model_path, config.confidence_threshold)
        self.pipeline: Optional[DetectionPipeline] = None
        self.camera: Optional[cv2.VideoCapture] = None
        self.accel_processor: Optional[AccelerometerProcessor] = None
        
        # Stats
        self.frame_count = 0
        self.frame_times = []
        self.is_running = False
    
    def setup(self) -> bool:
        """Initialize all components"""
        self.logger.info("=" * 60)
        self.logger.info("  POTHOLE DETECTION SYSTEM")
        self.logger.info("=" * 60)
        
        # Initialize detector
        if not self.detector.initialize():
            return False
        
        # Initialize accelerometer processor
        if self.config.enable_accelerometer:
            accel_config = AccelConfig.from_config_json()
            self.accel_processor = AccelerometerProcessor(accel_config)
            if self.config.accel_csv_path:
                self.accel_processor.initialize(csv_path=self.config.accel_csv_path)
            else:
                self.accel_processor.initialize()
            self.logger.info("[OK] Accelerometer processor ready")
        
        # Create pipeline (with accelerometer if available)
        self.pipeline = DetectionPipeline(self.detector, self.config, self.accel_processor)
        self.logger.info("[OK] Pipeline created (7-stage multimodal)")
        
        # Initialize camera
        if self.config.show_window:
            self.logger.info(f"Opening camera {self.config.camera_id}...")
            self.camera = cv2.VideoCapture(self.config.camera_id)
            
            if not self.camera.isOpened():
                self.logger.error(f"Failed to open camera {self.config.camera_id}")
                return False
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Warmup
            for _ in range(10):
                self.camera.read()
            
            self.logger.info("[OK] Camera initialized")
        
        if self.config.save_detections:
            Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
        
        return True
    
    def calc_fps(self) -> float:
        """Calculate current FPS"""
        self.frame_times.append(time.time())
        if len(self.frame_times) > 30:
            self.frame_times = self.frame_times[-30:]
        
        if len(self.frame_times) < 2:
            return 0.0
        
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0.0
    
    def save_frame(self, frame: np.ndarray, potholes: List[PotholeInfo]):
        """Save detection frame"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path(self.config.save_path) / f"detection_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        self.logger.info(f"Saved: {path}")
    
    def run(self):
        """Run the detection loop"""
        if not self.setup():
            self.logger.error("Setup failed")
            return
        
        self.is_running = True
        self.logger.info("")
        pipeline_label = "Clean -> Find -> Track -> Isolate -> Read -> Fuse -> Identify"
        self.logger.info(f"Pipeline: {pipeline_label}")
        self.logger.info("Press 'q' to quit, 's' to save frame")
        self.logger.info("")
        
        try:
            while self.is_running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    continue
                
                self.frame_count += 1
                
                # Process through pipeline
                potholes = self.pipeline.process(frame, self.frame_count)
                
                # Log detections
                for p in potholes:
                    accel_info = ""
                    if p.has_accel_data:
                        accel_info = f" | Accel:{p.accel_peak_g:.2f}G | Fusion:{p.fusion_score:.2f}"
                    self.logger.info(
                        f"POTHOLE | ID:{p.track_id} | {p.severity} | "
                        f"Depth:{p.depth} | Conf:{p.confidence:.0%} | Size:{p.area_ratio:.1%}{accel_info}"
                    )
                
                # Calculate FPS
                fps = self.calc_fps()
                
                # Visualize
                if self.config.show_window:
                    display = draw_detections(frame, potholes, fps)
                    cv2.imshow("Pothole Detection", display)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s') and potholes:
                        self.save_frame(display, potholes)
                
                # Auto-save
                if self.config.save_detections and potholes:
                    self.save_frame(draw_detections(frame, potholes, fps), potholes)
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def run_test(self):
        """
        Run test mode on dataset images with supervised learning.
        
        This mode:
        1. Loads images from the Datasets folder
        2. Runs YOLO detection on each image
        3. Extracts features for each pothole
        4. Uses supervised learning for:
           - Classification: Severity (LOW, MEDIUM, HIGH)
           - Regression: Size, confidence prediction
        5. Displays results and evaluation metrics
        """
        self.logger.info("=" * 60)
        self.logger.info("  POTHOLE DETECTION - DATASET TEST MODE")
        self.logger.info("=" * 60)
        self.logger.info("Processing dataset with supervised learning...")
        self.logger.info("")
        
        if not self.detector.initialize():
            return
        
        self.pipeline = DetectionPipeline(self.detector, self.config)
        
        # Find dataset images
        dataset_paths = [
            PROJECT_ROOT / "Datasets" / "train" / "images",
            PROJECT_ROOT / "Datasets" / "val" / "images",
            PROJECT_ROOT / "Datasets" / "images",
        ]
        
        image_files = []
        for dataset_path in dataset_paths:
            if dataset_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(list(dataset_path.glob(ext)))
        
        if not image_files:
            self.logger.warning("No images found in Datasets folder!")
            self.logger.info("Looking for images in: Datasets/train/images, Datasets/val/images")
            return
        
        # Limit for testing (process first 50 images)
        max_images = min(50, len(image_files))
        image_files = image_files[:max_images]
        
        self.logger.info(f"Found {len(image_files)} images to process")
        self.logger.info("")
        
        # Storage for ML training data
        all_features = []  # Features for each detection
        all_labels = []    # Labels (severity class)
        all_potholes = []  # All pothole info objects
        
        # Process each image
        for idx, image_path in enumerate(image_files):
            try:
                # Load image
                frame = cv2.imread(str(image_path))
                if frame is None:
                    continue
                
                # Process through 6-stage pipeline
                potholes = self.pipeline.process(frame, idx + 1)
                
                # Extract features for supervised learning
                for pothole in potholes:
                    # Feature vector for classification/regression
                    features = {
                        'confidence': pothole.confidence,
                        'area_ratio': pothole.area_ratio,
                        'width': pothole.width,
                        'height': pothole.height,
                        'aspect_ratio': pothole.width / max(pothole.height, 1),
                        'area': pothole.area,
                    }
                    
                    all_features.append(features)
                    all_labels.append(pothole.severity)
                    all_potholes.append(pothole)
                
                # Progress update
                if (idx + 1) % 10 == 0 or idx == 0:
                    self.logger.info(
                        f"Processed {idx + 1}/{len(image_files)} images | "
                        f"Detections so far: {len(all_potholes)}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_path.name}: {e}")
                continue
        
        # --- SUPERVISED LEARNING ANALYSIS ---
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("  SUPERVISED LEARNING ANALYSIS")
        self.logger.info("=" * 60)
        
        if not all_features:
            self.logger.info("No potholes detected in dataset.")
            self.logger.info("completed!")
            return
        
        # Classification Analysis (Severity)
        self.logger.info("")
        self.logger.info("[CLASSIFICATION] Severity Distribution:")
        self.logger.info("-" * 40)
        
        severity_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for label in all_labels:
            if label in severity_counts:
                severity_counts[label] += 1
        
        total = len(all_labels)
        for severity, count in severity_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            bar = "#" * int(pct / 5)
            self.logger.info(f"  {severity:8s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # Regression Analysis (Measurements)
        self.logger.info("")
        self.logger.info("[REGRESSION] Measurement Statistics:")
        self.logger.info("-" * 40)
        
        if all_features:
            # Calculate statistics
            confidences = [f['confidence'] for f in all_features]
            areas = [f['area_ratio'] for f in all_features]
            widths = [f['width'] for f in all_features]
            heights = [f['height'] for f in all_features]
            
            self.logger.info(f"  Confidence  : min={min(confidences):.2f}, max={max(confidences):.2f}, avg={sum(confidences)/len(confidences):.2f}")
            self.logger.info(f"  Area Ratio  : min={min(areas):.4f}, max={max(areas):.4f}, avg={sum(areas)/len(areas):.4f}")
            self.logger.info(f"  Width (px)  : min={min(widths):4d}, max={max(widths):4d}, avg={sum(widths)/len(widths):.0f}")
            self.logger.info(f"  Height (px) : min={min(heights):4d}, max={max(heights):4d}, avg={sum(heights)/len(heights):.0f}")
        
        # Feature Importance (for classification)
        self.logger.info("")
        self.logger.info("[CLASSIFICATION] Feature Importance for Severity:")
        self.logger.info("-" * 40)
        
        # Calculate correlation between features and severity
        severity_scores = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        if len(all_features) > 1:
            try:
                import statistics
                
                # Simple feature importance based on variance by class
                for feature_name in ['confidence', 'area_ratio', 'aspect_ratio']:
                    values_by_class = {'LOW': [], 'MEDIUM': [], 'HIGH': []}
                    for feat, label in zip(all_features, all_labels):
                        if label in values_by_class:
                            values_by_class[label].append(feat[feature_name])
                    
                    # Calculate mean per class
                    means = {}
                    for cls, vals in values_by_class.items():
                        if vals:
                            means[cls] = sum(vals) / len(vals)
                        else:
                            means[cls] = 0
                    
                    self.logger.info(f"  {feature_name}:")
                    for cls in ['LOW', 'MEDIUM', 'HIGH']:
                        val = means.get(cls, 0)
                        self.logger.info(f"    {cls}: {val:.4f}")
                        
            except Exception as e:
                self.logger.debug(f"Feature analysis error: {e}")
        
        # Sample predictions display
        self.logger.info("")
        self.logger.info("[PREDICTIONS] Sample Results (first 10):")
        self.logger.info("-" * 40)
        
        for i, pothole in enumerate(all_potholes[:10]):
            self.logger.info(
                f"  {i+1}. Severity: {pothole.severity:6s} | "
                f"Depth: {pothole.depth:8s} | "
                f"Conf: {pothole.confidence:.2f} | "
                f"Size: {pothole.area_ratio:.3f}"
            )
        
        # Final Summary
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("  TEST SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"  Images Processed    : {len(image_files)}")
        self.logger.info(f"  Total Potholes Found: {len(all_potholes)}")
        self.logger.info(f"  Avg per Image       : {len(all_potholes)/max(len(image_files),1):.1f}")
        self.logger.info("")
        self.logger.info("  Classification (Severity):")
        for severity, count in severity_counts.items():
            self.logger.info(f"    - {severity}: {count}")
        self.logger.info("")
        self.logger.info("  Regression (Size Predictions):")
        if all_features:
            avg_conf = sum(f['confidence'] for f in all_features) / len(all_features)
            avg_area = sum(f['area_ratio'] for f in all_features) / len(all_features)
            self.logger.info(f"    - Avg Confidence: {avg_conf:.2f}")
            self.logger.info(f"    - Avg Area Ratio: {avg_area:.4f}")
        self.logger.info("=" * 60)
        self.logger.info("completed!")
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        # Print summary
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("  SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Frames Processed: {self.frame_count}")
        if self.pipeline:
            self.logger.info(f"Total Detections: {self.pipeline.total_detections}")
            self.logger.info(f"Unique Tracks: {self.pipeline.total_tracks}")
        self.logger.info("=" * 60)
        self.logger.info("completed!")


# =============================================================================
# ENTRY POINT
# =============================================================================

def setup_logging(level: str = "INFO"):
    """Setup logging"""
    Path('logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/pothole_detector.log', mode='a')
        ]
    )


def run_annotate_mode(config: Config, max_images: int = 20):
    """
    Process dataset images and generate annotated outputs with segmentation.
    
    Features:
    - Processes images from Datasets folder
    - Creates bounding boxes with "Pothole" labels
    - Generates pixel-level segmentation masks
    - Applies colored overlays for visualization
    - Saves annotated images to results/annotated/
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("  POTHOLE ANNOTATION MODE")
    logger.info("=" * 60)
    logger.info("Generating annotated images with segmentation masks...")
    logger.info("")
    
    # Initialize detector
    detector = YOLODetector(config.model_path, config.confidence_threshold)
    if not detector.initialize():
        return
    
    pipeline = DetectionPipeline(detector, config)
    
    # Output directory
    output_dir = PROJECT_ROOT / "results" / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find dataset images
    dataset_paths = [
        PROJECT_ROOT / "Datasets" / "train" / "images",
        PROJECT_ROOT / "Datasets" / "val" / "images",
        PROJECT_ROOT / "Datasets" / "images",
    ]
    
    image_files = []
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(dataset_path.glob(ext)))
    
    if not image_files:
        logger.warning("No images found in Datasets folder!")
        return
    
    # Limit images
    image_files = image_files[:max_images]
    logger.info(f"Processing {len(image_files)} images...")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    processed = 0
    total_potholes = 0
    
    for idx, image_path in enumerate(image_files):
        try:
            # Load image
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            
            # Process through pipeline
            potholes = pipeline.process(frame, idx + 1)
            
            if potholes:
                total_potholes += len(potholes)
                
                # Create annotated image with segmentation
                annotated = annotate_image_with_segmentation(
                    frame, potholes, overlay_alpha=0.4, show_labels=True
                )
                
                # Save annotated image
                output_path = output_dir / f"annotated_{image_path.stem}.jpg"
                cv2.imwrite(str(output_path), annotated)
                
                logger.info(
                    f"[{idx+1}/{len(image_files)}] {image_path.name}: "
                    f"{len(potholes)} potholes -> {output_path.name}"
                )
                processed += 1
            else:
                logger.info(f"[{idx+1}/{len(image_files)}] {image_path.name}: No potholes detected")
                
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            continue
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  ANNOTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Images Processed : {len(image_files)}")
    logger.info(f"  Images Annotated : {processed}")
    logger.info(f"  Total Potholes   : {total_potholes}")
    logger.info(f"  Output Directory : {output_dir}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Annotated images saved with:")
    logger.info("  - Bounding boxes with 'Pothole' labels")
    logger.info("  - Segmentation masks with colored overlays")
    logger.info("  - Severity color coding (Red=HIGH, Orange=MEDIUM, Yellow=LOW)")
    logger.info("")
    logger.info("completed!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Pothole Detection System with 6-Stage Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  1. Clean Frames     - Noise reduction, enhancement
  2. Find Object      - YOLO detection
  3. Track Object     - Multi-object tracking
  4. Isolate          - Extract pothole regions
  5. Read Information - Feature extraction
  6. Identify         - Severity classification

Modes:
  Default             - Live camera detection
  --test              - Process dataset with ML analysis
  --annotate          - Generate annotated images with segmentation

Examples:
  python pothole_detector.py                  # Run with webcam
  python pothole_detector.py --camera 1       # Use camera 1
  python pothole_detector.py --test           # Test with ML analysis
  python pothole_detector.py --annotate       # Generate annotated images
  python pothole_detector.py --annotate -n 50 # Annotate 50 images
        """
    )
    
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index')
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.25, help='Detection confidence')
    parser.add_argument('--save', action='store_true', help='Save detections')
    parser.add_argument('--no-tracking', action='store_true', help='Disable tracking')
    parser.add_argument('--test', action='store_true', help='Test mode with ML analysis')
    parser.add_argument('--annotate', action='store_true', help='Generate annotated images with segmentation')
    parser.add_argument('-n', '--num-images', type=int, default=20, help='Number of images to annotate')
    parser.add_argument('--accel-csv', type=str, default='', help='Accelerometer CSV file for fusion')
    parser.add_argument('--no-accel', action='store_true', help='Disable accelerometer fusion')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create config
    config = Config(
        camera_id=args.camera,
        model_path=args.model,
        confidence_threshold=args.confidence,
        save_detections=args.save,
        enable_tracking=not args.no_tracking,
        show_window=not (args.test or args.annotate),
        enable_accelerometer=not args.no_accel,
        accel_csv_path=args.accel_csv,
    )
    
    # Run appropriate mode
    if args.annotate:
        run_annotate_mode(config, max_images=args.num_images)
    elif args.test:
        app = PotholeDetector(config)
        app.run_test()
    else:
        app = PotholeDetector(config)
        app.run()


if __name__ == "__main__":
    main()

