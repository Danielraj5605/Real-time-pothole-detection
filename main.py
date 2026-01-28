"""
Main Application Entry Point
Demonstrates complete system with JSON configuration and dependency injection
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.application.config import get_config, get_container
from src.application.events import PotholeDetectedEvent, AlertTriggeredEvent
from src.infrastructure.persistence.adapters import SQLiteRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main application entry point"""
    
    logger.info("=" * 70)
    logger.info("POTHOLE DETECTION SYSTEM - JSON-CONFIGURED ARCHITECTURE")
    logger.info("=" * 70)
    
    # ========================================================================
    # 1. LOAD CONFIGURATION (JSON ONLY)
    # ========================================================================
    logger.info("\n[1] Loading JSON configuration...")
    config = get_config()
    
    logger.info(f"   Environment: {config.get('environment', 'development')}")
    logger.info(f"   Hardware Mode: {config.get('hardware.mode', 'mock')}")
    logger.info(f"   Vision Model: {config.get('vision.model_type', 'yolov8n')}")
    logger.info(f"   Detection Confidence: {config.get('detection.min_confidence', 0.5)}")
    logger.info("✓ Configuration loaded from JSON")
    
    # ========================================================================
    # 2. INITIALIZE DEPENDENCY CONTAINER
    # ========================================================================
    logger.info("\n[2] Initializing dependency injection container...")
    container = get_container()
    
    # Get all services (lazy initialization)
    detection_service = container.get_detection_service()
    alert_service = container.get_alert_service()
    reporting_service = container.get_reporting_service()
    event_bus = container.get_event_bus()
    
    logger.info("✓ All services initialized via dependency injection")
    
    # ========================================================================
    # 3. INITIALIZE PERSISTENCE LAYER
    # ========================================================================
    logger.info("\n[3] Initializing persistence layer...")
    repository = SQLiteRepository(
        database_path=config.get('persistence.database_path', 'data/database/potholes.db')
    )
    logger.info("✓ SQLite repository initialized")
    
    # ========================================================================
    # 4. REGISTER EVENT HANDLERS
    # ========================================================================
    logger.info("\n[4] Registering event handlers...")
    
    detected_potholes = []
    triggered_alerts = []
    
    async def on_pothole_detected(event: PotholeDetectedEvent):
        """Handle pothole detection event"""
        pothole = event.pothole
        detected_potholes.append(pothole)
        
        # Save to database
        repository.save_pothole(pothole)
        
        logger.info(f"\n📍 POTHOLE DETECTED AND SAVED:")
        logger.info(f"   ID: {pothole.id}")
        logger.info(f"   Severity: {pothole.severity.value}")
        logger.info(f"   Confidence: {pothole.confidence:.2f}")
        logger.info(f"   Location: ({pothole.latitude:.6f}, {pothole.longitude:.6f})")
        logger.info(f"   Acceleration: {pothole.accel_peak:.2f}g")
    
    async def on_alert_triggered(event: AlertTriggeredEvent):
        """Handle alert trigger event"""
        alert = event.alert
        triggered_alerts.append(alert)
        
        # Save to database
        repository.save_alert(alert)
        
        logger.info(f"   Alert saved to database: {alert.id}")
    
    event_bus.subscribe(PotholeDetectedEvent, on_pothole_detected)
    event_bus.subscribe(AlertTriggeredEvent, on_alert_triggered)
    
    logger.info("✓ Event handlers registered")
    
    # ========================================================================
    # 5. START EVENT BUS
    # ========================================================================
    logger.info("\n[5] Starting event bus...")
    bus_task = asyncio.create_task(event_bus.start())
    logger.info("✓ Event bus running")
    
    # ========================================================================
    # 6. RUN DETECTION PIPELINE
    # ========================================================================
    logger.info("\n[6] Running detection pipeline...")
    logger.info(f"Processing {config.get('demo.frame_count', 20)} frames...\n")
    
    frame_count = 20
    for i in range(frame_count):
        logger.info(f"Frame {i+1}/{frame_count}")
        
        # Process one frame
        pothole = await detection_service.process_frame()
        
        # Check for proximity alerts
        if detected_potholes:
            gps = container.get_gps()
            gps_reading = gps.read()
            
            await alert_service.check_proximity(
                current_lat=gps_reading.data.latitude,
                current_lon=gps_reading.data.longitude,
                known_potholes=detected_potholes
            )
        
        # Simulate frame rate
        await asyncio.sleep(1.0 / config.get('detection.frame_rate', 15))
    
    # ========================================================================
    # 7. GENERATE REPORTS
    # ========================================================================
    logger.info("\n[7] Generating reports...")
    
    if detected_potholes:
        summary = reporting_service.generate_summary(detected_potholes)
        
        logger.info("\n" + "=" * 70)
        logger.info("DETECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Potholes Detected: {summary['total_count']}")
        logger.info(f"  - HIGH severity: {summary['by_severity']['HIGH']}")
        logger.info(f"  - MEDIUM severity: {summary['by_severity']['MEDIUM']}")
        logger.info(f"  - LOW severity: {summary['by_severity']['LOW']}")
        logger.info(f"Average Confidence: {summary['average_confidence']:.2f}")
        logger.info(f"Average Acceleration: {summary['average_acceleration']:.2f}g")
        logger.info(f"Alerts Triggered: {len(triggered_alerts)}")
        logger.info(f"Verified Potholes: {summary['verified_count']}")
        logger.info("=" * 70)
        
        # Database statistics
        all_potholes = repository.get_all_potholes()
        logger.info(f"\nDatabase contains {len(all_potholes)} total potholes")
        
        # Geographic report
        geo_report = reporting_service.generate_geographic_report(
            detected_potholes,
            grid_size_deg=0.001
        )
        
        if geo_report:
            logger.info("\nGEOGRAPHIC DISTRIBUTION:")
            for cell in geo_report[:5]:
                logger.info(
                    f"  Grid ({cell['lat_min']:.4f}, {cell['lon_min']:.4f}): "
                    f"{cell['count']} potholes "
                    f"(H:{cell['high_severity']}, M:{cell['medium_severity']}, L:{cell['low_severity']})"
                )
    else:
        logger.info("\nNo potholes detected in this run.")
    
    # ========================================================================
    # 8. CLEANUP
    # ========================================================================
    logger.info("\n[8] Cleaning up resources...")
    
    event_bus.stop()
    await bus_task
    
    repository.cleanup()
    container.cleanup()
    
    logger.info("✓ All resources released")
    logger.info("\n" + "=" * 70)
    logger.info("APPLICATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"\nConfiguration: config/config.json")
    logger.info(f"Database: {config.get('persistence.database_path')}")
    logger.info(f"Logs: {config.get('logging.log_file')}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user")
    except Exception as e:
        logger.error(f"\nApplication error: {e}", exc_info=True)
        sys.exit(1)
