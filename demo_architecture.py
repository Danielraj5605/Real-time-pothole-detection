"""
Example: Using the New Layered Architecture with Mock Sensors

This demonstrates the complete detection pipeline using the new architecture.
No hardware required - uses mock sensors for testing.
"""
import asyncio
import logging
from pathlib import Path

# Infrastructure layer
from src.infrastructure.sensors.adapters import MockCamera, MockAccelerometer, MockGPS
from src.infrastructure.ml.adapters import MockDetector

# Domain layer
from src.domain.services.fusion_service import FusionService
from src.domain.services.proximity_calculator import ProximityCalculator

# Application layer
from src.application.services.detection_service import DetectionService, DetectionConfig
from src.application.services.alert_service import AlertService
from src.application.services.reporting_service import ReportingService
from src.application.events.event_bus import EventBus
from src.application.events.pothole_detected import PotholeDetectedEvent
from src.application.events.alert_triggered import AlertTriggeredEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConsoleAlertChannel:
    """Simple console alert channel for demonstration"""
    
    def send_alert(self, alert):
        """Print alert to console"""
        print(f"\n🚨 ALERT: {alert.level.value}")
        print(f"   Message: {alert.message}")
        print(f"   Distance: {alert.distance_meters:.1f}m\n")
        return True


async def main():
    """Main demonstration function"""
    
    logger.info("=" * 60)
    logger.info("Pothole Detection System - Architecture Demo")
    logger.info("=" * 60)
    
    # ========================================================================
    # 1. INFRASTRUCTURE LAYER - Initialize hardware adapters
    # ========================================================================
    logger.info("\n[1] Initializing infrastructure layer...")
    
    camera = MockCamera(width=640, height=480)
    accelerometer = MockAccelerometer()
    gps = MockGPS(start_lat=37.7749, start_lon=-122.4194)  # San Francisco
    detector = MockDetector(detection_probability=0.4)
    
    # Initialize all sensors
    camera.initialize()
    camera.calibrate()
    
    accelerometer.initialize()
    accelerometer.calibrate()
    accelerometer.enable_pothole_mode(True)  # Simulate potholes
    
    gps.initialize()
    gps.calibrate()
    
    detector.initialize()
    
    logger.info("✓ All sensors initialized")
    
    # ========================================================================
    # 2. DOMAIN LAYER - Create domain services
    # ========================================================================
    logger.info("\n[2] Creating domain services...")
    
    fusion_service = FusionService(
        vision_weight=0.6,
        accel_weight=0.4,
        fusion_threshold=0.5
    )
    
    proximity_calculator = ProximityCalculator()
    
    logger.info("✓ Domain services created")
    
    # ========================================================================
    # 3. APPLICATION LAYER - Create application services
    # ========================================================================
    logger.info("\n[3] Creating application services...")
    
    event_bus = EventBus()
    
    # Detection service configuration
    detection_config = DetectionConfig(
        min_confidence=0.5,
        accel_threshold=1.5,
        frame_rate=15,
        vision_weight=0.6,
        accel_weight=0.4,
        fusion_threshold=0.5
    )
    
    detection_service = DetectionService(
        camera=camera,
        accelerometer=accelerometer,
        gps=gps,
        detector=detector,
        fusion_service=fusion_service,
        event_bus=event_bus,
        config=detection_config
    )
    
    # Alert service with console channel
    alert_channels = [ConsoleAlertChannel()]
    alert_service = AlertService(
        proximity_calculator=proximity_calculator,
        alert_channels=alert_channels,
        event_bus=event_bus,
        max_distance_m=200
    )
    
    # Reporting service
    reporting_service = ReportingService()
    
    logger.info("✓ Application services created")
    
    # ========================================================================
    # 4. EVENT HANDLERS - Subscribe to events
    # ========================================================================
    logger.info("\n[4] Setting up event handlers...")
    
    detected_potholes = []
    triggered_alerts = []
    
    async def on_pothole_detected(event: PotholeDetectedEvent):
        """Handle pothole detection event"""
        pothole = event.pothole
        detected_potholes.append(pothole)
        
        logger.info(f"\n📍 POTHOLE DETECTED:")
        logger.info(f"   ID: {pothole.id}")
        logger.info(f"   Severity: {pothole.severity.value}")
        logger.info(f"   Confidence: {pothole.confidence:.2f}")
        logger.info(f"   Location: ({pothole.latitude:.6f}, {pothole.longitude:.6f})")
        logger.info(f"   Acceleration: {pothole.accel_peak:.2f}g")
    
    async def on_alert_triggered(event: AlertTriggeredEvent):
        """Handle alert trigger event"""
        alert = event.alert
        triggered_alerts.append(alert)
    
    event_bus.subscribe(PotholeDetectedEvent, on_pothole_detected)
    event_bus.subscribe(AlertTriggeredEvent, on_alert_triggered)
    
    logger.info("✓ Event handlers registered")
    
    # ========================================================================
    # 5. START SYSTEM - Begin processing
    # ========================================================================
    logger.info("\n[5] Starting detection system...")
    logger.info("Processing 20 frames...\n")
    
    # Start event bus in background
    bus_task = asyncio.create_task(event_bus.start())
    
    # Process frames
    for i in range(20):
        logger.info(f"Frame {i+1}/20")
        
        # Process one frame
        pothole = await detection_service.process_frame()
        
        # Check for proximity alerts if we have detected potholes
        if detected_potholes:
            gps_reading = gps.read()
            await alert_service.check_proximity(
                current_lat=gps_reading.data.latitude,
                current_lon=gps_reading.data.longitude,
                known_potholes=detected_potholes
            )
        
        # Simulate frame rate
        await asyncio.sleep(1.0 / detection_config.frame_rate)
    
    # ========================================================================
    # 6. GENERATE REPORTS - Show statistics
    # ========================================================================
    logger.info("\n[6] Generating reports...")
    
    if detected_potholes:
        summary = reporting_service.generate_summary(detected_potholes)
        
        logger.info("\n" + "=" * 60)
        logger.info("DETECTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Potholes Detected: {summary['total_count']}")
        logger.info(f"  - HIGH severity: {summary['by_severity']['HIGH']}")
        logger.info(f"  - MEDIUM severity: {summary['by_severity']['MEDIUM']}")
        logger.info(f"  - LOW severity: {summary['by_severity']['LOW']}")
        logger.info(f"Average Confidence: {summary['average_confidence']:.2f}")
        logger.info(f"Average Acceleration: {summary['average_acceleration']:.2f}g")
        logger.info(f"Alerts Triggered: {len(triggered_alerts)}")
        logger.info("=" * 60)
        
        # Geographic report
        geo_report = reporting_service.generate_geographic_report(
            detected_potholes,
            grid_size_deg=0.001
        )
        
        if geo_report:
            logger.info("\nGEOGRAPHIC DISTRIBUTION:")
            for cell in geo_report[:5]:  # Show top 5
                logger.info(
                    f"  Grid ({cell['lat_min']:.4f}, {cell['lon_min']:.4f}): "
                    f"{cell['count']} potholes"
                )
    else:
        logger.info("\nNo potholes detected in this run.")
    
    # ========================================================================
    # 7. CLEANUP - Release resources
    # ========================================================================
    logger.info("\n[7] Cleaning up...")
    
    event_bus.stop()
    await bus_task
    
    camera.cleanup()
    accelerometer.cleanup()
    gps.cleanup()
    detector.cleanup()
    
    logger.info("✓ All resources released")
    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
