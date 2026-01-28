"""
Dependency Injection Container
Manages creation and lifecycle of all system components
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .settings import get_config

# Infrastructure imports
from ...infrastructure.sensors.interfaces import (
    CameraInterface, AccelerometerInterface, GPSInterface
)
from ...infrastructure.sensors.adapters import (
    OpenCVCamera, MPU6050Accelerometer, NEO6MGPS,
    MockCamera, MockAccelerometer, MockGPS
)
from ...infrastructure.ml.interfaces import DetectorInterface
from ...infrastructure.ml.adapters import YOLOv8Detector, MockDetector

# Domain imports
from ...domain.services import FusionService, ProximityCalculator, SeverityClassifier

# Application imports
from ..events.event_bus import EventBus
from ..services import DetectionService, AlertService, ReportingService, DetectionConfig


class DependencyContainer:
    """
    Dependency Injection Container.
    Creates and manages all system components based on configuration.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        self._instances: Dict[str, Any] = {}
    
    def get_camera(self) -> CameraInterface:
        """Get camera instance based on configuration"""
        if 'camera' not in self._instances:
            mode = self.config.get('hardware.mode', 'mock')
            
            if mode == 'mock':
                camera = MockCamera(
                    width=self.config.get('hardware.camera.width', 640),
                    height=self.config.get('hardware.camera.height', 480)
                )
            else:
                camera = OpenCVCamera(
                    camera_id=self.config.get('hardware.camera.device_id', 0)
                )
            
            camera.initialize()
            camera.calibrate()
            self._instances['camera'] = camera
            self.logger.info(f"Camera initialized: {type(camera).__name__}")
        
        return self._instances['camera']
    
    def get_accelerometer(self) -> AccelerometerInterface:
        """Get accelerometer instance based on configuration"""
        if 'accelerometer' not in self._instances:
            mode = self.config.get('hardware.mode', 'mock')
            
            if mode == 'mock':
                accel = MockAccelerometer()
            else:
                accel = MPU6050Accelerometer(
                    i2c_address=int(self.config.get('accelerometer.sensor.i2c_address', '0x68'), 16),
                    i2c_bus=self.config.get('accelerometer.sensor.i2c_bus', 1)
                )
            
            accel.initialize()
            accel.calibrate()
            
            # Set range if real hardware
            if mode != 'mock':
                accel.set_range(self.config.get('accelerometer.sensor.accelerometer_range_g', 2))
            
            self._instances['accelerometer'] = accel
            self.logger.info(f"Accelerometer initialized: {type(accel).__name__}")
        
        return self._instances['accelerometer']
    
    def get_gps(self) -> GPSInterface:
        """Get GPS instance based on configuration"""
        if 'gps' not in self._instances:
            mode = self.config.get('hardware.mode', 'mock')
            
            if mode == 'mock' or self.config.get('gps.simulation_mode', True):
                gps = MockGPS(
                    start_lat=self.config.get('gps.default_location.latitude', 40.4533),
                    start_lon=self.config.get('gps.default_location.longitude', -79.9463)
                )
            else:
                gps = NEO6MGPS(
                    port=self.config.get('gps.sensor.port', '/dev/ttyAMA0'),
                    baudrate=self.config.get('gps.sensor.baudrate', 9600)
                )
            
            gps.initialize()
            gps.calibrate()
            self._instances['gps'] = gps
            self.logger.info(f"GPS initialized: {type(gps).__name__}")
        
        return self._instances['gps']
    
    def get_detector(self) -> DetectorInterface:
        """Get ML detector instance based on configuration"""
        if 'detector' not in self._instances:
            mode = self.config.get('hardware.mode', 'mock')
            
            if mode == 'mock':
                detector = MockDetector(detection_probability=0.4)
            else:
                model_path = self.config.get('vision.trained_weights', 'yolov8n.pt')
                detector = YOLOv8Detector(
                    model_path=model_path,
                    confidence_threshold=self.config.get('vision.inference.confidence_threshold', 0.25),
                    iou_threshold=self.config.get('vision.inference.iou_threshold', 0.45)
                )
            
            detector.initialize()
            self._instances['detector'] = detector
            self.logger.info(f"Detector initialized: {type(detector).__name__}")
        
        return self._instances['detector']
    
    def get_fusion_service(self) -> FusionService:
        """Get fusion service instance"""
        if 'fusion_service' not in self._instances:
            fusion = FusionService(
                vision_weight=self.config.get('fusion.rule_based.vision_weight', 0.6),
                accel_weight=self.config.get('fusion.rule_based.accel_weight', 0.4),
                fusion_threshold=self.config.get('fusion.rule_based.combined_detection_threshold', 0.5)
            )
            self._instances['fusion_service'] = fusion
            self.logger.info("Fusion service created")
        
        return self._instances['fusion_service']
    
    def get_severity_classifier(self) -> SeverityClassifier:
        """Get severity classifier instance"""
        if 'severity_classifier' not in self._instances:
            classifier = SeverityClassifier(
                high_accel_threshold=self.config.get('severity.high.accel_threshold', 2.5),
                high_confidence_threshold=self.config.get('severity.high.confidence_threshold', 0.7),
                high_bbox_threshold=self.config.get('severity.high.bbox_threshold', 10000),
                medium_accel_threshold=self.config.get('severity.medium.accel_threshold', 1.8),
                medium_confidence_threshold=self.config.get('severity.medium.confidence_threshold', 0.6),
                medium_bbox_threshold=self.config.get('severity.medium.bbox_threshold', 5000)
            )
            self._instances['severity_classifier'] = classifier
            self.logger.info("Severity classifier created")
        
        return self._instances['severity_classifier']
    
    def get_proximity_calculator(self) -> ProximityCalculator:
        """Get proximity calculator instance"""
        if 'proximity_calculator' not in self._instances:
            calc = ProximityCalculator()
            self._instances['proximity_calculator'] = calc
            self.logger.info("Proximity calculator created")
        
        return self._instances['proximity_calculator']
    
    def get_event_bus(self) -> EventBus:
        """Get event bus instance"""
        if 'event_bus' not in self._instances:
            bus = EventBus()
            self._instances['event_bus'] = bus
            self.logger.info("Event bus created")
        
        return self._instances['event_bus']
    
    def get_detection_service(self) -> DetectionService:
        """Get detection service instance"""
        if 'detection_service' not in self._instances:
            detection_config = DetectionConfig(
                min_confidence=self.config.get('detection.min_confidence', 0.5),
                accel_threshold=self.config.get('detection.accel_threshold', 1.5),
                frame_rate=self.config.get('detection.frame_rate', 15),
                vision_weight=self.config.get('detection.vision_weight', 0.6),
                accel_weight=self.config.get('detection.accel_weight', 0.4),
                fusion_threshold=self.config.get('detection.fusion_threshold', 0.5)
            )
            
            service = DetectionService(
                camera=self.get_camera(),
                accelerometer=self.get_accelerometer(),
                gps=self.get_gps(),
                detector=self.get_detector(),
                fusion_service=self.get_fusion_service(),
                event_bus=self.get_event_bus(),
                config=detection_config
            )
            self._instances['detection_service'] = service
            self.logger.info("Detection service created")
        
        return self._instances['detection_service']
    
    def get_alert_service(self) -> AlertService:
        """Get alert service instance"""
        if 'alert_service' not in self._instances:
            # Import alert channels here to avoid circular imports
            from ...infrastructure.alerts.adapters import ConsoleAlertChannel
            
            channels = []
            if self.config.get('alerts.channels.console.enabled', True):
                channels.append(ConsoleAlertChannel())
            
            service = AlertService(
                proximity_calculator=self.get_proximity_calculator(),
                alert_channels=channels,
                event_bus=self.get_event_bus(),
                max_distance_m=self.config.get('alerts.max_distance_m', 200)
            )
            self._instances['alert_service'] = service
            self.logger.info("Alert service created")
        
        return self._instances['alert_service']
    
    def get_reporting_service(self) -> ReportingService:
        """Get reporting service instance"""
        if 'reporting_service' not in self._instances:
            service = ReportingService()
            self._instances['reporting_service'] = service
            self.logger.info("Reporting service created")
        
        return self._instances['reporting_service']
    
    def get_repository(self):
        """Get persistence repository instance"""
        if 'repository' not in self._instances:
            from ...infrastructure.persistence.adapters import SQLiteRepository
            
            db_path = self.config.get('persistence.database_path', 'data/database/potholes.db')
            repository = SQLiteRepository(db_path)
            # Repository auto-initializes in __init__
            
            self._instances['repository'] = repository
            self.logger.info(f"Repository initialized: SQLite at {db_path}")
        
        return self._instances['repository']
    
    def cleanup(self):
        """Cleanup all resources (sync version)"""
        self.logger.info("Cleaning up resources...")
        
        # Cleanup sensors
        for key in ['camera', 'accelerometer', 'gps', 'detector', 'repository']:
            if key in self._instances:
                try:
                    self._instances[key].cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up {key}: {e}")
        
        # Stop event bus
        if 'event_bus' in self._instances:
            try:
                self._instances['event_bus'].stop()
            except Exception as e:
                self.logger.error(f"Error stopping event bus: {e}")
        
        self._instances.clear()
        self.logger.info("Cleanup complete")
    
    async def cleanup_async(self):
        """Cleanup all resources (async version)"""
        self.logger.info("Cleaning up resources...")
        
        # Cleanup sensors
        for key in ['camera', 'accelerometer', 'gps', 'detector', 'repository']:
            if key in self._instances:
                try:
                    self._instances[key].cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up {key}: {e}")
        
        # Stop event bus
        if 'event_bus' in self._instances:
            try:
                await self._instances['event_bus'].stop()
            except Exception as e:
                self.logger.error(f"Error stopping event bus: {e}")
        
        self._instances.clear()
        self.logger.info("Cleanup complete")


# Global container instance
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """Get global dependency container instance"""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container
