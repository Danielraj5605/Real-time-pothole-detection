#!/usr/bin/env python3
"""
Pothole Detection Dataset Collector
Main script for synchronized multi-sensor data collection on Raspberry Pi Zero WH.

Collects:
- Video from OV5647 camera (via rpicam-vid subprocess)
- Accelerometer/Gyroscope data from MPU9250 (100Hz)
- GPS data from NEO-6M (1Hz) - OPTIONAL
- Manual pothole labels (via keyboard in SSH terminal)

Usage:
    python collect_data.py [options]
    
Options:
    --duration SECONDS   Recording duration (default: unlimited, Ctrl+C to stop)
    --output PATH        Output directory (default: ~/pothole_dataset/data/session_YYYYMMDD_HHMMSS)
    --no-video           Disable video recording (sensor data only)
    --no-gps             Disable GPS completely (for when GPS module is unavailable)
    --calibrate          Run accelerometer calibration before recording
"""

import os
import sys
import time
import json
import csv
import subprocess
import threading
import signal
import select
import termios
import tty
from datetime import datetime
from pathlib import Path

# Import our sensor drivers
from mpu9250_driver import MPU9250

# GPS is optional - import with fallback
try:
    from gps_driver import GPS
    GPS_AVAILABLE = True
except ImportError:
    GPS_AVAILABLE = False
    print("Note: GPS driver not available, GPS features disabled")


# ==============================================================================
# THREAD-SAFE ACCELEROMETER STATE (for real-time access)
# ==============================================================================

class ThreadSafeAccelState:
    """
    Thread-safe container storing ONLY the latest accelerometer sample.
    This eliminates buffering delay - reader always gets the most recent data.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {'accel': (0, 0, 0), 'gyro': (0, 0, 0), 'timestamp_ms': 0}
    
    def update(self, accel, gyro, timestamp_ms):
        """Update with new sample (called by sensor thread at high frequency)."""
        with self._lock:
            self._data = {'accel': accel, 'gyro': gyro, 'timestamp_ms': timestamp_ms}
    
    def get_latest(self):
        """Get the most recent sample (called by main thread)."""
        with self._lock:
            return self._data.copy()


class DataCollector:
    """
    Main data collection orchestrator.
    Synchronizes camera, accelerometer, and GPS data collection.
    """
    
    def __init__(self, output_dir=None, enable_video=True, disable_gps=False):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Output directory path (created if doesn't exist)
            enable_video: Whether to record video
            disable_gps: Completely disable GPS (for when GPS module is unavailable)
        """
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path.home() / "pothole_dataset" / "data" / f"session_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "video").mkdir(exist_ok=True)
        (self.output_dir / "accelerometer").mkdir(exist_ok=True)
        if not disable_gps:
            (self.output_dir / "gps").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        
        self.enable_video = enable_video
        self.enable_gps = not disable_gps and GPS_AVAILABLE
        
        # Sensors
        self.mpu = None
        self.gps = None
        
        # Video process
        self.video_process = None
        
        # Data files
        self.accel_file = None
        self.accel_writer = None
        self.gps_file = None
        self.gps_writer = None
        
        # Labels
        self.labels = []
        
        # Control flags
        self.running = False
        
        # Thread-safe accelerometer state for real-time access
        self.realtime_accel = ThreadSafeAccelState()
        self.start_time_ms = 0
        
        # Unified timestamp reference (T0) using high-resolution monotonic clock
        # All timestamps (video PTS, accelerometer) are relative to this T0
        self.T0_ns = 0  # perf_counter_ns at video process start
        self.T0_epoch = 0  # Corresponding epoch time for reference
        
        # Session metadata
        self.metadata = {
            "session_id": self.output_dir.name,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "total_accel_samples": 0,
            "total_gps_samples": 0,
            "total_labels": 0,
            "sensor_config": {
                "camera": {
                    "resolution": "1280x960",
                    "framerate": 30,
                    "codec": "h264",
                    "bitrate": 8000000
                },
                "accelerometer": {
                    "sample_rate_hz": 100,
                    "accel_range_g": 16,
                    "gyro_range_dps": 2000
                },
                "gps": {
                    "sample_rate_hz": 1
                }
            },
            "video_start_offset_ms": 0,  # Offset between video and sensor timestamps
            
            # Unified timestamp sync fields (Phase 1)
            "T0_ns": 0,  # perf_counter_ns at video process start
            "T0_epoch": 0,  # Epoch time corresponding to T0
            "camera_start_ns": 0,  # Estimated first frame capture time (relative to T0)
            "accel_start_ns": 0  # When accelerometer sampling started (relative to T0)
        }
    
    def initialize_sensors(self, calibrate_accel=False):
        """
        Initialize all sensors.
        
        Args:
            calibrate_accel: Whether to run accelerometer calibration
            
        Returns:
            True if all sensors initialized, False otherwise
        """
        print("\n" + "=" * 50)
        print("INITIALIZING SENSORS")
        print("=" * 50)
        
        # Initialize MPU9250
        print("\n[1/2] Initializing MPU9250 accelerometer...")
        try:
            self.mpu = MPU9250(accel_range=16, gyro_range=2000)
            if not self.mpu.verify_connection():
                print("WARNING: Could not verify MPU9250, but will try to use it")
            
            if calibrate_accel:
                print("\nCalibration mode - keep sensor flat and still!")
                time.sleep(2)
                self.mpu.calibrate(samples=100)
            else:
                print("Skipping calibration (use --calibrate flag to enable)")
            
            # Test read
            test_data = self.mpu.read_all()
            print(f"Test read: accel={test_data['accel'][2]:.2f}g (should be ~1.0g if flat)")
            print("MPU9250 ready!")
        except Exception as e:
            print(f"ERROR: Failed to initialize MPU9250: {e}")
            return False
        
        # Initialize GPS (optional)
        if self.enable_gps:
            print("\n[2/2] Initializing GPS...")
            try:
                self.gps = GPS()
                if not self.gps.connect():
                    print("WARNING: Failed to connect to GPS, continuing without GPS")
                    self.enable_gps = False
                    self.gps = None
                else:
                    self.gps.start_reading()
                    print("GPS initialized (not waiting for fix)")
                    print("GPS data will be recorded when fix is available")
            except Exception as e:
                print(f"WARNING: GPS initialization failed: {e}")
                print("Continuing without GPS...")
                self.enable_gps = False
                self.gps = None
        else:
            print("\n[2/2] GPS disabled (--no-gps flag or module unavailable)")
        
        print("\n" + "=" * 50)
        print("SENSORS INITIALIZED")
        print(f"  MPU9250: OK")
        print(f"  GPS: {'OK' if self.enable_gps else 'DISABLED'}")
        print("=" * 50)
        return True
    
    def start_video_recording(self):
        """Start video recording using rpicam-vid subprocess."""
        if not self.enable_video:
            print("Video recording disabled")
            return True
        
        video_path = self.output_dir / "video" / "raw_capture.h264"
        pts_path = self.output_dir / "video" / "frame_timestamps.txt"
        
        cmd = [
            "rpicam-vid",
            "-t", "0",  # Record until stopped
            "--width", "1280",
            "--height", "960",
            "--framerate", "30",
            "--codec", "h264",
            "--bitrate", "8000000",
            "--inline",  # Include SPS/PPS headers
            "--profile", "baseline",  # Low latency profile
            "--save-pts", str(pts_path),  # Save frame timestamps!
            "-o", str(video_path)
        ]
        
        print(f"\nStarting video recording: {video_path}")
        try:
            # ============================================================
            # UNIFIED TIMESTAMP: Record T0 at exact process start
            # ============================================================
            # T0 is our single time reference for synchronizing:
            # - Video PTS (relative to first frame capture)
            # - Accelerometer timestamps (relative to T0)
            self.T0_ns = time.perf_counter_ns()
            self.T0_epoch = time.time()
            process_start_time = self.T0_epoch  # Keep for backward compat
            
            self.video_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for camera AND encoder to stabilize
            # Note: First frame is captured ~100-200ms after process start
            # PTS=0 corresponds to that first frame, NOT to T0
            print("Waiting for camera and encoder to stabilize (3 seconds)...")
            time.sleep(3)
            
            if self.video_process.poll() is not None:
                print("ERROR: rpicam-vid failed to start")
                return False
            
            # Record stabilization end time (relative to T0)
            stabilization_end_ns = time.perf_counter_ns()
            
            # Estimate when camera started capturing first frame
            # Camera typically needs ~100-200ms to warm up after process start
            # PTS=0 corresponds to first frame, which is at T0 + camera_warmup
            estimated_camera_warmup_ns = 150_000_000  # ~150ms typical
            self.metadata["camera_start_ns"] = estimated_camera_warmup_ns
            self.metadata["stabilization_duration_ns"] = stabilization_end_ns - self.T0_ns
            
            # Save T0 reference in metadata
            self.metadata["T0_ns"] = self.T0_ns
            self.metadata["T0_epoch"] = self.T0_epoch
            
            # Legacy fields for backward compatibility
            self.video_start_time = time.time()
            self.metadata["video_process_start_epoch"] = process_start_time
            self.metadata["video_recording_start_epoch"] = self.video_start_time
            print(f"Video recording started! (PTS file: {pts_path})")
            print(f"  T0 recorded at epoch {self.T0_epoch:.3f}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to start video recording: {e}")
            return False
    
    def stop_video_recording(self):
        """Stop video recording."""
        if self.video_process:
            print("Stopping video recording...")
            self.video_process.terminate()
            try:
                self.video_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.video_process.kill()
            self.video_process = None
            print("Video recording stopped")
    
    def open_data_files(self):
        """Open CSV files for accelerometer and GPS data."""
        # Accelerometer CSV
        accel_path = self.output_dir / "accelerometer" / "accel_log.csv"
        self.accel_file = open(accel_path, 'w', newline='')
        self.accel_writer = csv.writer(self.accel_file)
        self.accel_writer.writerow([
            'timestamp_ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'
        ])
        
        # GPS CSV (only if GPS enabled)
        if self.enable_gps:
            gps_path = self.output_dir / "gps" / "gps_log.csv"
            self.gps_file = open(gps_path, 'w', newline='')
            self.gps_writer = csv.writer(self.gps_file)
            self.gps_writer.writerow([
                'timestamp_ms', 'latitude', 'longitude', 'altitude_m',
                'speed_kmh', 'heading_deg', 'hdop', 'fix_quality', 'satellites'
            ])
        
        print(f"Data files opened in: {self.output_dir}")
    
    def close_data_files(self):
        """Close all data files."""
        if self.accel_file:
            self.accel_file.close()
            self.accel_file = None
        
        if self.gps_file:
            self.gps_file.close()
            self.gps_file = None
    
    def get_timestamp_ns(self):
        """Get current timestamp in nanoseconds since T0 (unified reference)."""
        return time.perf_counter_ns() - self.T0_ns
    
    def get_timestamp_ms(self):
        """Get current timestamp in milliseconds since T0 (unified reference)."""
        return self.get_timestamp_ns() // 1_000_000
    
    def accel_sampling_thread(self):
        """Background thread for accelerometer sampling at ~100Hz."""
        sample_interval = 0.01  # 10ms = 100Hz
        next_sample_time = time.time()
        sample_count = 0
        
        while self.running:
            try:
                # Read sensor data
                data = self.mpu.read_all()
                timestamp = self.get_timestamp_ms()
                
                ax, ay, az = data['accel']
                gx, gy, gz = data['gyro']
                
                # Update real-time state (for instant access by main thread)
                self.realtime_accel.update((ax, ay, az), (gx, gy, gz), timestamp)
                
                # Write to CSV
                self.accel_writer.writerow([
                    timestamp,
                    f"{ax:.4f}", f"{ay:.4f}", f"{az:.4f}",
                    f"{gx:.2f}", f"{gy:.2f}", f"{gz:.2f}"
                ])
                
                sample_count += 1
                
                # Flush periodically
                if sample_count % 100 == 0:
                    self.accel_file.flush()
                
                # Sleep until next sample time
                next_sample_time += sample_interval
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Reset if we're behind
                    next_sample_time = time.time()
            
            except Exception as e:
                print(f"\nAccel error: {e}")
                time.sleep(0.1)
        
        self.metadata["total_accel_samples"] = sample_count
    
    def gps_sampling_thread(self):
        """Background thread for GPS sampling at 1Hz."""
        sample_count = 0
        
        while self.running:
            try:
                data = self.gps.get_all()
                timestamp = self.get_timestamp_ms()
                
                self.gps_writer.writerow([
                    timestamp,
                    f"{data['latitude']:.6f}" if data['latitude'] else "",
                    f"{data['longitude']:.6f}" if data['longitude'] else "",
                    f"{data['altitude_m']:.1f}",
                    f"{data['speed_kmh']:.1f}",
                    f"{data['heading_deg']:.1f}",
                    f"{data['hdop']:.1f}",
                    data['fix_quality'],
                    data['satellites']
                ])
                
                sample_count += 1
                self.gps_file.flush()
                
                time.sleep(1.0)
            
            except Exception as e:
                print(f"\nGPS error: {e}")
                time.sleep(1.0)
        
        self.metadata["total_gps_samples"] = sample_count
    
    def add_label(self, event_type="pothole_label"):
        """Add a label event at current timestamp."""
        timestamp = self.get_timestamp_ms()
        
        event = {
            "timestamp_ms": timestamp,
            "event_type": event_type,
            "gps": self.gps.get_all() if self.gps else None
        }
        
        self.labels.append(event)
        print(f"\n>>> LABEL ADDED at {timestamp}ms ({len(self.labels)} total) <<<")
    
    def save_labels(self):
        """Save labels to JSON file."""
        labels_path = self.output_dir / "labels" / "events.json"
        
        with open(labels_path, 'w') as f:
            json.dump({"events": self.labels}, f, indent=2)
        
        self.metadata["total_labels"] = len(self.labels)
        print(f"Saved {len(self.labels)} labels to {labels_path}")
    
    def save_metadata(self):
        """Save session metadata."""
        self.metadata["end_time"] = datetime.now().isoformat()
        
        if self.metadata["start_time"]:
            start = datetime.fromisoformat(self.metadata["start_time"])
            end = datetime.fromisoformat(self.metadata["end_time"])
            self.metadata["duration_seconds"] = (end - start).total_seconds()
        
        metadata_path = self.output_dir / "metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Saved metadata to {metadata_path}")
    
    def run(self, duration=None):
        """
        Run data collection.
        
        Args:
            duration: Recording duration in seconds (None for unlimited)
        """
        print("\n" + "=" * 50)
        print("STARTING DATA COLLECTION")
        print("=" * 50)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Duration: {'unlimited (Ctrl+C to stop)' if duration is None else f'{duration} seconds'}")
        print("\n" + "-" * 50)
        print("CONTROLS:")
        print("  [SPACE] or [ENTER] - Mark pothole label")
        print("  [Ctrl+C]           - Stop recording")
        print("-" * 50)
        
        # Open data files
        self.open_data_files()
        
        # Start video FIRST
        if not self.start_video_recording():
            print("WARNING: Video recording failed, continuing with sensors only")
            self.enable_video = False
        
        # Record start time AFTER video is running and stabilized
        # This ensures accelerometer timestamps start from NOW, not from video start
        self.metadata["start_time"] = datetime.now().isoformat()
        
        # ============================================================
        # UNIFIED TIMESTAMP: Record when accelerometer sampling starts
        # ============================================================
        # Accel timestamps are relative to T0 (set in start_video_recording)
        # Record accel_start_ns so we know the offset between first frame and first accel sample
        self.metadata["accel_start_ns"] = self.get_timestamp_ns()
        print(f"Accelerometer starting at T0 + {self.metadata['accel_start_ns'] // 1_000_000}ms")
        
        # Legacy field for backward compatibility
        self.start_time_ms = time.time() * 1000
        
        # Calculate video offset (how many seconds of video exist before accel started)
        if hasattr(self, 'video_start_time') and self.video_start_time:
            video_offset_ms = int(self.start_time_ms - (self.video_start_time * 1000))
            self.metadata["video_start_offset_ms"] = video_offset_ms
            print(f"Video started {video_offset_ms}ms before accelerometer")
        else:
            self.metadata["video_start_offset_ms"] = 0
        
        self.running = True
        
        # Start sensor threads AFTER video is confirmed running
        accel_thread = threading.Thread(target=self.accel_sampling_thread, daemon=True)
        accel_thread.start()
        
        gps_thread = None
        if self.enable_gps:
            gps_thread = threading.Thread(target=self.gps_sampling_thread, daemon=True)
            gps_thread.start()
        
        print("\n>>> RECORDING STARTED <<<")
        print("Press [SPACE] when you encounter a pothole!\n")
        
        # Set up terminal for non-blocking input
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            start_time = time.time()
            last_status_time = 0
            
            while self.running:
                # Check for keyboard input (non-blocking)
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    if key == ' ' or key == '\n' or key == '\r':
                        self.add_label()
                    elif key == '\x03':  # Ctrl+C
                        print("\n\nStopping...")
                        break
                
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    print(f"\n\nDuration limit ({duration}s) reached.")
                    break
                
                # Status update every 5 seconds
                current_time = time.time()
                if current_time - last_status_time >= 5:
                    elapsed = int(current_time - start_time)
                    gps_data = self.gps.get_all() if self.gps else {}
                    
                    # Use real-time accelerometer state (no I2C call, instant access)
                    accel_data = self.realtime_accel.get_latest()
                    
                    # Calculate acceleration magnitude
                    ax, ay, az = accel_data['accel']
                    accel_mag = (ax**2 + ay**2 + az**2) ** 0.5
                    
                    # Build status string
                    status = f"[{elapsed:4d}s] Labels: {len(self.labels):3d} | Accel: {accel_mag:.2f}g"
                    
                    if self.enable_gps and self.gps:
                        gps_data = self.gps.get_all()
                        status += f" | GPS: {'FIX' if self.gps.has_fix() else 'NO FIX'} ({gps_data.get('satellites', 0)} sats)"
                        status += f" | Speed: {gps_data.get('speed_kmh', 0):.1f} km/h"
                    else:
                        status += " | GPS: DISABLED"
                    
                    print(status)
                    
                    last_status_time = current_time
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            # Stop everything - MUST be in finally block to ensure cleanup on Ctrl+C
            self.running = False
            
            print("\nStopping sensors...")
            accel_thread.join(timeout=2)
            if gps_thread:
                gps_thread.join(timeout=2)
            
            self.stop_video_recording()
            self.close_data_files()
            self.save_labels()
            self.save_metadata()
        
        # Print summary
        print("\n" + "=" * 50)
        print("DATA COLLECTION COMPLETE")
        print("=" * 50)
        print(f"\nSession: {self.metadata['session_id']}")
        print(f"Duration: {self.metadata['duration_seconds']:.1f} seconds")
        print(f"Accelerometer samples: {self.metadata['total_accel_samples']}")
        print(f"GPS samples: {self.metadata['total_gps_samples']}")
        print(f"Pothole labels: {self.metadata['total_labels']}")
        print(f"\nData saved to: {self.output_dir}")
        print("\nTo copy to your laptop:")
        print(f"  scp -r raspberrypi@<IP>:{self.output_dir} .")
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.stop_video_recording()
        self.close_data_files()
        
        if self.mpu:
            self.mpu.close()
            self.mpu = None
        
        if self.gps:
            self.gps.close()
            self.gps = None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pothole Detection Dataset Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_data.py                     # Start recording (Ctrl+C to stop)
  python collect_data.py --duration 300      # Record for 5 minutes
  python collect_data.py --calibrate         # Calibrate accelerometer first
  python collect_data.py --no-gps            # Skip GPS fix wait (for testing)
  python collect_data.py --no-video          # Sensor data only (no video)
        """
    )
    
    parser.add_argument('--duration', type=int, default=None,
                        help='Recording duration in seconds (default: unlimited)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory path')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--no-gps', action='store_true',
                        help='Skip waiting for GPS fix')
    parser.add_argument('--calibrate', action='store_true',
                        help='Run accelerometer calibration before recording')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("  POTHOLE DETECTION DATASET COLLECTOR")
    print("  Raspberry Pi Zero WH Edition")
    print("=" * 50)
    
    collector = DataCollector(
        output_dir=args.output,
        enable_video=not args.no_video,
        disable_gps=args.no_gps
    )
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        collector.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize sensors
        if not collector.initialize_sensors(calibrate_accel=args.calibrate):
            print("\nFailed to initialize sensors. Exiting.")
            sys.exit(1)
        
        # Run data collection
        collector.run(duration=args.duration)
    
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
