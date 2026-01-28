"""
Real-Time Pothole Detection System - Main Entry Point

Demonstrates the complete modular event-driven layered architecture.
Supports both mock mode (testing) and real mode (live camera).

Usage:
    python main.py              # Run with mock sensors (demo mode)
    python main.py --live       # Run with live camera detection
"""
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the application."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'pothole_detection.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from JSON files."""
    config_path = project_root / "config" / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


async def run_mock_demo(config: dict, logger: logging.Logger):
    """Run demonstration with mock sensors."""
    logger.info("=" * 60)
    logger.info("  POTHOLE DETECTION SYSTEM - MOCK DEMO")
    logger.info("=" * 60)
    
    # Import components
    from src.application.config.dependency_injection import DependencyContainer, get_container
    import asyncio
    
    logger.info("Initializing components...")
    
    # Get or create container (uses global)
    container = get_container()
    
    # Get event bus from container and start as background task
    event_bus = container.get_event_bus()
    event_bus_task = asyncio.create_task(event_bus.start())
    
    logger.info("[OK] Components initialized")
    logger.info(f"Configuration: config/config.json")
    logger.info(f"Hardware mode: {config.get('hardware', {}).get('mode', 'mock')}")
    
    # Get repository
    repository = container.get_repository()
    logger.info(f"Database: {config.get('persistence', {}).get('database_path', 'data/database/potholes.db')}")
    
    # Get services
    detection_service = container.get_detection_service()
    alert_service = container.get_alert_service()
    reporting_service = container.get_reporting_service()
    
    logger.info("[OK] Services created")
    logger.info("")
    logger.info("Running mock detection pipeline...")
    logger.info("-" * 40)
    
    # Run detection cycles
    detected_potholes = []
    num_cycles = 10
    
    for i in range(num_cycles):
        pothole = await detection_service.process_frame()
        
        if pothole:
            detected_potholes.append(pothole)
            logger.info(
                f"[Cycle {i+1}] Pothole detected: "
                f"severity={pothole.severity.value}, "
                f"confidence={pothole.confidence:.2f}"
            )
            
            # Save to database
            try:
                repository.save_pothole(pothole)
            except Exception as e:
                logger.warning(f"Failed to save pothole: {e}")
        else:
            logger.debug(f"[Cycle {i+1}] No pothole detected")
        
        await asyncio.sleep(0.1)
    
    logger.info("-" * 40)
    logger.info(f"Detection complete: {len(detected_potholes)} potholes detected in {num_cycles} cycles")
    
    # Generate report
    logger.info("")
    logger.info("Generating report...")
    report = reporting_service.generate_summary(detected_potholes)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("  SESSION REPORT")
    logger.info("=" * 60)
    logger.info(f"Total Detections: {report.get('total_detections', 0)}")
    severity_breakdown = report.get('by_severity', {})
    logger.info(f"  - HIGH: {severity_breakdown.get('HIGH', 0)}")
    logger.info(f"  - MEDIUM: {severity_breakdown.get('MEDIUM', 0)}")
    logger.info(f"  - LOW: {severity_breakdown.get('LOW', 0)}")
    logger.info("=" * 60)
    
    # Cleanup
    logger.info("")
    logger.info("Cleaning up...")
    
    # Stop event bus task
    event_bus.stop()
    event_bus_task.cancel()
    try:
        await event_bus_task
    except asyncio.CancelledError:
        pass
    
    container.cleanup()
    
    logger.info("[OK] Cleanup complete")
    logger.info("")
    logger.info("=" * 60)
    logger.info("  DEMO COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Run 'python live_detection.py' for live camera detection")
    logger.info("  - Edit 'config/config.json' to customize settings")
    logger.info("  - See 'ARCHITECTURE.md' for system documentation")


def run_live_detection(config: dict, logger: logging.Logger, args: argparse.Namespace):
    """Run live camera detection."""
    logger.info("Starting live camera detection...")
    logger.info("Launching live_detection.py...")
    
    import subprocess
    cmd = [sys.executable, "live_detection.py"]
    
    if args.camera is not None:
        cmd.extend(["--camera", str(args.camera)])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.confidence:
        cmd.extend(["--confidence", str(args.confidence)])
    if args.save:
        cmd.append("--save")
    if args.no_display:
        cmd.append("--no-display")
    
    subprocess.run(cmd, cwd=str(project_root))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Real-Time Pothole Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Default (no args)  : Run mock demo to test the architecture
  --live             : Run live camera detection

Examples:
  python main.py                    # Mock demo
  python main.py --live             # Live camera
  python main.py --live --camera 1  # Live with camera 1
        """
    )
    
    parser.add_argument(
        '--live', '-l',
        action='store_true',
        help='Enable live camera detection'
    )
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=None,
        help='Camera device index (for --live mode)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to YOLO model weights (for --live mode)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=None,
        help='Detection confidence threshold (for --live mode)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save detection frames (for --live mode)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without display window (for --live mode)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Load configuration
        config = load_config()
        
        if args.live:
            # Run live detection
            run_live_detection(config, logger, args)
        else:
            # Run mock demo
            asyncio.run(run_mock_demo(config, logger))
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
