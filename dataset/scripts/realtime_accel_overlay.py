#!/usr/bin/env python3
"""
Real-Time Accelerometer Overlay on Camera Feed
Low-latency solution for Raspberry Pi (Pi Zero / Pi 4)

This script demonstrates how to minimize latency between accelerometer
readings and camera frames by:
1. Reading accelerometer at high frequency (≥100Hz) in a separate thread
2. Storing ONLY the latest sample (no buffering delay)
3. Using timestamp-based synchronization
4. Overlaying values immediately after frame capture

Usage:
    python realtime_accel_overlay.py [options]

Options:
    --debug          Show timing/latency information
    --benchmark      Run 100-frame benchmark and exit
    --filter ALPHA   Apply IIR low-pass filter (default: no filter, 0.0-1.0)
    --headless       No display, just print stats (for SSH testing)
    --sample-rate HZ Accelerometer sample rate (default: 200)
    
Example:
    python realtime_accel_overlay.py --debug --filter 0.2
"""

import time
import threading
import argparse
from dataclasses import dataclass, field
from typing import Optional

# ==============================================================================
# ACCELEROMETER SAMPLE DATACLASS
# ==============================================================================

@dataclass
class AccelSample:
    """
    Immutable container for a single accelerometer/gyroscope reading.
    Using a dataclass ensures clean, type-safe data passing.
    """
    # Accelerometer values in g (gravitational units)
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    
    # Gyroscope values in degrees per second
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    
    # High-precision timestamp (nanoseconds since start)
    timestamp_ns: int = 0
    
    # Sample counter for debugging dropped samples
    sample_count: int = 0
    
    def magnitude(self) -> float:
        """Calculate total acceleration magnitude."""
        return (self.ax**2 + self.ay**2 + self.az**2) ** 0.5


# ==============================================================================
# THREAD-SAFE ACCELEROMETER STATE
# ==============================================================================

class ThreadSafeAccelState:
    """
    Thread-safe container that stores ONLY the latest accelerometer sample.
    
    WHY THIS DESIGN:
    - No queue = no buffering delay
    - Lock is held briefly (just for copy) = minimal contention
    - Reader always gets the most recent data
    
    This is the KEY to eliminating the ~2 second delay you were seeing.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._sample: AccelSample = AccelSample()
        self._start_time_ns = time.perf_counter_ns()
    
    def update(self, sample: AccelSample) -> None:
        """
        Update with new sample. Called by sensor thread at high frequency.
        Lock is held only for the assignment (nanoseconds).
        """
        with self._lock:
            self._sample = sample
    
    def get_latest(self) -> AccelSample:
        """
        Get the most recent sample. Called by camera thread.
        Returns a copy to avoid race conditions.
        """
        with self._lock:
            return self._sample
    
    def get_start_time_ns(self) -> int:
        """Get the reference start time for timestamp calculations."""
        return self._start_time_ns


# ==============================================================================
# LOW-LATENCY IIR FILTER (OPTIONAL)
# ==============================================================================

class LowLatencyFilter:
    """
    Single-pole IIR (Infinite Impulse Response) low-pass filter.
    
    WHY NOT MOVING AVERAGE:
    - Moving average of N samples has delay of (N-1)/2 samples
    - At 100Hz with N=10: delay = 45ms
    
    WHY IIR FILTER:
    - Group delay ≈ (1-α)/α samples at low frequencies
    - α=0.2: delay ≈ 4 samples = 40ms at 100Hz, but...
    - For step response, 63% reached in 1/(α) samples = 5 samples = 50ms
    - α=0.5: much more responsive (10ms to 63%), slight noise
    
    RECOMMENDATION:
    - For visual display: α=0.2 to 0.3 (smooth, ~30-40ms delay)
    - For pothole detection: α=0.5 or no filter (responsive)
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize filter.
        
        Args:
            alpha: Filter coefficient (0.0-1.0)
                   Higher = more responsive, more noise
                   Lower = smoother, more delay
                   0.0 = filter disabled (pass-through)
        """
        self.alpha = alpha
        self._prev: Optional[AccelSample] = None
    
    def filter(self, sample: AccelSample) -> AccelSample:
        """Apply filter to sample. First sample passes through unchanged."""
        if self.alpha <= 0.0 or self.alpha >= 1.0 or self._prev is None:
            self._prev = sample
            return sample
        
        # IIR filter: filtered = α * new + (1-α) * previous
        a = self.alpha
        b = 1.0 - a
        
        filtered = AccelSample(
            ax = a * sample.ax + b * self._prev.ax,
            ay = a * sample.ay + b * self._prev.ay,
            az = a * sample.az + b * self._prev.az,
            gx = a * sample.gx + b * self._prev.gx,
            gy = a * sample.gy + b * self._prev.gy,
            gz = a * sample.gz + b * self._prev.gz,
            timestamp_ns = sample.timestamp_ns,
            sample_count = sample.sample_count
        )
        
        self._prev = filtered
        return filtered


# ==============================================================================
# HIGH-FREQUENCY ACCELEROMETER READER THREAD
# ==============================================================================

class AccelReader:
    """
    Background thread that reads the accelerometer at high frequency.
    
    KEY DESIGN POINTS:
    1. Uses precise timing (not sleep(0.01) which drifts)
    2. Updates shared state with each new sample
    3. Never blocks the camera thread
    4. Handles I2C errors gracefully
    """
    
    def __init__(self, state: ThreadSafeAccelState, sample_rate_hz: int = 200,
                 filter_alpha: float = 0.0):
        """
        Initialize the accelerometer reader.
        
        Args:
            state: Shared thread-safe state for samples
            sample_rate_hz: Target sample rate (default 200Hz for low latency)
            filter_alpha: Optional IIR filter coefficient (0 = disabled)
        """
        self.state = state
        self.sample_rate_hz = sample_rate_hz
        self.sample_interval = 1.0 / sample_rate_hz
        self.filter = LowLatencyFilter(filter_alpha) if filter_alpha > 0 else None
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mpu = None
        self._sample_count = 0
        
        # Statistics for debugging
        self.stats = {
            'total_samples': 0,
            'read_errors': 0,
            'avg_read_time_us': 0,
            'max_read_time_us': 0,
        }
    
    def start(self) -> bool:
        """
        Initialize sensor and start the sampling thread.
        
        Returns:
            True if started successfully, False otherwise
        """
        # Import and initialize MPU9250
        try:
            from mpu9250_driver import MPU9250
            self._mpu = MPU9250(accel_range=16, gyro_range=2000)
            
            if not self._mpu.verify_connection():
                print("WARNING: Could not verify MPU9250, attempting anyway...")
            
            # Test read
            test = self._mpu.read_all()
            print(f"MPU9250 initialized: Z-accel = {test['accel'][2]:.2f}g")
            
        except ImportError:
            print("ERROR: mpu9250_driver.py not found!")
            print("Make sure it's in the same directory.")
            return False
        except Exception as e:
            print(f"ERROR: Failed to initialize MPU9250: {e}")
            return False
        
        # Start sampling thread
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        
        print(f"Accelerometer reader started at {self.sample_rate_hz}Hz")
        return True
    
    def stop(self):
        """Stop the sampling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._mpu:
            self._mpu.close()
    
    def _sample_loop(self):
        """
        Main sampling loop. Runs in background thread.
        
        TIMING STRATEGY:
        - Calculate next_sample_time based on start time
        - Sleep until next_sample_time (if time remains)
        - If we're behind, catch up without sleeping
        - This prevents drift accumulation
        """
        start_time = time.perf_counter()
        next_sample_time = start_time
        total_read_time_ns = 0
        
        while self._running:
            try:
                # ===== READ SENSOR =====
                read_start = time.perf_counter_ns()
                data = self._mpu.read_all()
                read_end = time.perf_counter_ns()
                
                # Track timing statistics
                read_time_us = (read_end - read_start) // 1000
                total_read_time_ns += (read_end - read_start)
                self.stats['max_read_time_us'] = max(
                    self.stats['max_read_time_us'], read_time_us
                )
                
                # ===== CREATE SAMPLE =====
                self._sample_count += 1
                sample = AccelSample(
                    ax=data['accel'][0],
                    ay=data['accel'][1],
                    az=data['accel'][2],
                    gx=data['gyro'][0],
                    gy=data['gyro'][1],
                    gz=data['gyro'][2],
                    timestamp_ns=read_end - self.state.get_start_time_ns(),
                    sample_count=self._sample_count
                )
                
                # ===== APPLY FILTER (if enabled) =====
                if self.filter:
                    sample = self.filter.filter(sample)
                
                # ===== UPDATE SHARED STATE =====
                # This is the critical handoff to the camera thread
                self.state.update(sample)
                
                self.stats['total_samples'] = self._sample_count
                
            except Exception as e:
                self.stats['read_errors'] += 1
                if self.stats['read_errors'] <= 5:
                    print(f"Accel read error: {e}")
                time.sleep(0.01)  # Back off on error
                continue
            
            # ===== PRECISE TIMING =====
            # Calculate when the NEXT sample should occur
            next_sample_time += self.sample_interval
            
            # Sleep until next sample time (if we have time left)
            now = time.perf_counter()
            sleep_time = next_sample_time - now
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -self.sample_interval:
                # We're more than one sample behind - reset timing
                # This can happen after system hiccups
                next_sample_time = time.perf_counter()
        
        # Final statistics
        if self._sample_count > 0:
            self.stats['avg_read_time_us'] = (
                total_read_time_ns // self._sample_count // 1000
            )


# ==============================================================================
# OVERLAY DRAWING FUNCTIONS
# ==============================================================================

def draw_overlay(frame, sample: AccelSample, debug_info: dict = None):
    """
    Draw accelerometer overlay on camera frame.
    
    Args:
        frame: OpenCV image (modified in-place)
        sample: Current accelerometer sample
        debug_info: Optional dict with timing info for debug mode
    
    Returns:
        Modified frame
    """
    import cv2
    
    h, w = frame.shape[:2]
    magnitude = sample.magnitude()
    
    # ===== SEMI-TRANSPARENT BACKGROUND =====
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # ===== COLOR CODING =====
    white = (255, 255, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    
    # Magnitude color based on value
    if magnitude > 2.0:
        mag_color = red
    elif magnitude > 1.5:
        mag_color = yellow
    else:
        mag_color = green
    
    # ===== TEXT OVERLAY =====
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 35
    
    # Timestamp
    time_sec = sample.timestamp_ns / 1e9
    cv2.putText(frame, f"Time: {time_sec:.2f}s  Sample: {sample.sample_count}",
                (20, y), font, 0.5, white, 1)
    y += 25
    
    # Accelerometer values
    cv2.putText(frame, f"Accel X: {sample.ax:+.3f} g", (20, y), font, 0.55, white, 1)
    y += 22
    cv2.putText(frame, f"Accel Y: {sample.ay:+.3f} g", (20, y), font, 0.55, white, 1)
    y += 22
    cv2.putText(frame, f"Accel Z: {sample.az:+.3f} g", (20, y), font, 0.55, white, 1)
    y += 25
    
    # Magnitude (larger, colored)
    cv2.putText(frame, f"Magnitude: {magnitude:.3f} g", (20, y), font, 0.7, mag_color, 2)
    y += 25
    
    # ===== DEBUG TIMING INFO =====
    if debug_info:
        latency_ms = debug_info.get('latency_ms', 0)
        fps = debug_info.get('fps', 0)
        
        # Show latency with color coding
        if latency_ms < 20:
            lat_color = green
        elif latency_ms < 50:
            lat_color = yellow
        else:
            lat_color = red
        
        cv2.putText(frame, f"Latency: {latency_ms:.1f}ms  FPS: {fps:.1f}",
                    (20, y), font, 0.5, lat_color, 1)
    
    # ===== MAGNITUDE BAR =====
    bar_x, bar_y = 230, 40
    bar_w, bar_h = 80, 100
    
    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    
    # Fill based on magnitude (capped at 3g)
    fill_h = int(min(magnitude / 3.0, 1.0) * bar_h)
    fill_y = bar_y + bar_h - fill_h
    cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_w, bar_y + bar_h),
                  mag_color, -1)
    
    # Reference lines
    cv2.line(frame, (bar_x, bar_y + int(bar_h * 2/3)),
             (bar_x + bar_w, bar_y + int(bar_h * 2/3)), white, 1)
    cv2.putText(frame, "1g", (bar_x + bar_w + 5, bar_y + int(bar_h * 2/3) + 5),
                font, 0.4, white, 1)
    
    return frame


# ==============================================================================
# MAIN CAMERA LOOP
# ==============================================================================

def main():
    """
    Main entry point: Camera capture with real-time accelerometer overlay.
    """
    import cv2
    
    # ===== PARSE ARGUMENTS =====
    parser = argparse.ArgumentParser(
        description="Real-time accelerometer overlay on camera feed",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--debug', action='store_true',
                        help='Show timing/latency information')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run 100-frame benchmark and exit')
    parser.add_argument('--filter', type=float, default=0.0,
                        help='IIR filter alpha (0.0-1.0, default: 0 = no filter)')
    parser.add_argument('--headless', action='store_true',
                        help='No display window (for SSH)')
    parser.add_argument('--sample-rate', type=int, default=200,
                        help='Accelerometer sample rate Hz (default: 200)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("REAL-TIME ACCELEROMETER OVERLAY")
    print("=" * 50)
    print(f"Sample rate: {args.sample_rate}Hz")
    print(f"Filter: {'α=' + str(args.filter) if args.filter > 0 else 'disabled'}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print("=" * 50 + "\n")
    
    # ===== INITIALIZE SHARED STATE =====
    state = ThreadSafeAccelState()
    
    # ===== START ACCELEROMETER READER =====
    reader = AccelReader(
        state=state,
        sample_rate_hz=args.sample_rate,
        filter_alpha=args.filter
    )
    
    if not reader.start():
        print("Failed to start accelerometer reader!")
        return 1
    
    # ===== INITIALIZE CAMERA (using picamera2 for libcamera support) =====
    print("Initializing camera...")
    
    picam2 = None
    use_picamera2 = False
    
    try:
        from picamera2 import Picamera2
        
        picam2 = Picamera2()
        
        # Configure for low latency: small buffers, no post-processing
        config = picam2.create_preview_configuration(
            main={"size": (1280, 960), "format": "RGB888"},
            buffer_count=2,  # Minimal buffering for low latency
            queue=False      # Don't queue frames - always get latest
        )
        picam2.configure(config)
        picam2.start()
        
        # Let camera stabilize
        time.sleep(0.5)
        
        use_picamera2 = True
        print("Camera initialized with picamera2 (low-latency mode)")
        print("Camera: 1280x960")
        
    except ImportError:
        print("picamera2 not available, trying OpenCV...")
        # Fallback to OpenCV (may not work on Pi with libcamera)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera!")
            print("Install picamera2: sudo apt install -y python3-picamera2")
            reader.stop()
            return 1
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Camera opened with OpenCV")
    except Exception as e:
        print(f"ERROR: Camera initialization failed: {e}")
        print("Try: sudo apt install -y python3-picamera2")
        reader.stop()
        return 1
    
    # ===== TIMING VARIABLES =====
    frame_count = 0
    fps_start_time = time.perf_counter()
    fps = 0.0
    
    # Benchmark mode variables
    benchmark_latencies = []
    benchmark_frames = 100
    
    print("\nPress Ctrl+C to quit\n")
    
    # ===== MAIN LOOP =====
    try:
        while True:
            loop_start = time.perf_counter_ns()
            
            # ----- CAPTURE FRAME -----
            if use_picamera2:
                # picamera2: capture_array() gets the latest frame
                frame = picam2.capture_array()
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame!")
                    break
            
            capture_time = time.perf_counter_ns()
            
            # ----- GET LATEST ACCELEROMETER SAMPLE -----
            # This is always the most recent sample - no buffering!
            sample = state.get_latest()
            sample_time = time.perf_counter_ns()
            
            # ----- CALCULATE LATENCY -----
            # Time from when sample was captured to now
            sample_age_ns = sample_time - state.get_start_time_ns() - sample.timestamp_ns
            latency_ms = sample_age_ns / 1e6
            
            # ----- DRAW OVERLAY -----
            debug_info = None
            if args.debug or args.benchmark:
                debug_info = {
                    'latency_ms': latency_ms,
                    'fps': fps,
                }
            
            frame = draw_overlay(frame, sample, debug_info)
            
            # ----- DISPLAY (unless headless) -----
            if not args.headless:
                cv2.imshow('Accelerometer Overlay', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # ----- UPDATE FPS -----
            frame_count += 1
            elapsed = time.perf_counter() - fps_start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.perf_counter()
                
                if args.debug:
                    print(f"FPS: {fps:.1f}  Latency: {latency_ms:.1f}ms  "
                          f"Samples: {reader.stats['total_samples']}  "
                          f"Errors: {reader.stats['read_errors']}")
            
            # ----- BENCHMARK MODE -----
            if args.benchmark:
                benchmark_latencies.append(latency_ms)
                if len(benchmark_latencies) >= benchmark_frames:
                    break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # ===== CLEANUP =====
        reader.stop()
        if use_picamera2 and picam2:
            picam2.stop()
        elif 'cap' in dir() and cap:
            cap.release()
        cv2.destroyAllWindows()
    
    # ===== BENCHMARK RESULTS =====
    if args.benchmark and benchmark_latencies:
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Frames analyzed: {len(benchmark_latencies)}")
        print(f"Average latency: {sum(benchmark_latencies)/len(benchmark_latencies):.2f}ms")
        print(f"Min latency: {min(benchmark_latencies):.2f}ms")
        print(f"Max latency: {max(benchmark_latencies):.2f}ms")
        print(f"Avg sensor read time: {reader.stats['avg_read_time_us']}µs")
        print(f"Max sensor read time: {reader.stats['max_read_time_us']}µs")
        print("=" * 50)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
