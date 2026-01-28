"""
Detection Service - Orchestrates the Detection Pipeline
"""
from typing import Protocol, Optional
from dataclasses import dataclass
import logging
import numpy as np

from ...domain.entities.pothole import Pothole, Severity
from ...domain.services.fusion_service import FusionService
from ..events.event_bus import EventBus
from ..events.pothole_detected import PotholeDetectedEvent


# Ports (Dependency Inversion Principle)
class CameraPort(Protocol):
    """Port for camera - Dependency Inversion"""
    def capture_frame(self):
        """Capture a frame"""
        ...


class AccelerometerPort(Protocol):
    """Port for accelerometer"""
    def read(self):
        """Read acceleration data"""
        ...


class GPSPort(Protocol):
    """Port for GPS"""
    def read(self):
        """Read GPS coordinates"""
        ...


class DetectorPort(Protocol):
    """Port for ML detector"""
    def detect(self, image: np.ndarray) -> list:
        """Detect objects in image"""
        ...


@dataclass
class DetectionConfig:
    """Configuration for detection service"""
    min_confidence: float = 0.5
    accel_threshold: float = 1.5
    frame_rate: int = 15
    vision_weight: float = 0.6
    accel_weight: float = 0.4
    fusion_threshold: float = 0.5


class DetectionService:
    """
    Orchestrates the detection pipeline.
    Depends only on abstractions (DIP).
    """
    
    def __init__(
        self,
        camera: CameraPort,
        accelerometer: AccelerometerPort,
        gps: GPSPort,
        detector: DetectorPort,
        fusion_service: FusionService,
        event_bus: EventBus,
        config: DetectionConfig
    ):
        """
        Initialize detection service.
        
        Args:
            camera: Camera port implementation
            accelerometer: Accelerometer port implementation
            gps: GPS port implementation
            detector: ML detector port implementation
            fusion_service: Multimodal fusion service
            event_bus: Event bus for publishing events
            config: Detection configuration
        """
        self._camera = camera
        self._accelerometer = accelerometer
        self._gps = gps
        self._detector = detector
        self._fusion = fusion_service
        self._event_bus = event_bus
        self._config = config
        self._logger = logging.getLogger(__name__)
    
    async def process_frame(self) -> Optional[Pothole]:
        """
        Process a single frame through the detection pipeline.
        
        Returns:
            Pothole entity if detected, None otherwise
        """
        try:
            # 1. Capture sensor data
            frame_reading = self._camera.capture_frame()
            if frame_reading is None:
                self._logger.warning("Failed to capture frame")
                return None
            
            accel_reading = self._accelerometer.read()
            gps_reading = self._gps.read()
            
            # 2. Run ML detection
            detections = self._detector.detect(frame_reading.image)
            
            # Convert detections to dict format for fusion
            detection_dicts = [
                {
                    'confidence': d.confidence,
                    'bbox': d.bbox
                }
                for d in detections
            ]
            
            # 3. Get acceleration data
            accel_data = {
                'x': accel_reading.data.x,
                'y': accel_reading.data.y,
                'z': accel_reading.data.z
            }
            
            # 4. Apply multimodal fusion
            result = self._fusion.fuse(
                visual_detections=detection_dicts,
                acceleration_data=accel_data,
                min_confidence=self._config.min_confidence,
                accel_threshold=self._config.accel_threshold
            )
            
            if result.is_pothole_detected:
                # 5. Get GPS coordinates
                lat = gps_reading.data.latitude
                lon = gps_reading.data.longitude
                
                # 6. Create domain entity
                pothole = Pothole(
                    latitude=lat,
                    longitude=lon,
                    severity=Severity.from_metrics(
                        accel_peak=result.accel_peak,
                        confidence=result.confidence,
                        bbox_area=result.bbox_area
                    ),
                    confidence=result.confidence,
                    accel_peak=result.accel_peak,
                    bbox_area=result.bbox_area
                )
                
                # 7. Publish event (loosely coupled)
                await self._event_bus.publish(
                    PotholeDetectedEvent(
                        pothole=pothole,
                        raw_frame=frame_reading.image
                    )
                )
                
                self._logger.info(
                    f"Pothole detected: severity={pothole.severity.value}, "
                    f"confidence={pothole.confidence:.2f}"
                )
                
                return pothole
            
            return None
            
        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")
            return None
    
    def process_frame_sync(self) -> Optional[Pothole]:
        """Synchronous version of process_frame"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.process_frame())
        except RuntimeError:
            return asyncio.run(self.process_frame())
