"""
Process Live Session Data - Video + Accelerometer Fusion

This script processes real-world data collected from Raspberry Pi:
1. Detects pothole impacts from accelerometer data
2. Extracts corresponding video frames
3. Runs YOLO detection on frames
4. Applies vision + accelerometer fusion
5. Classifies pothole severity (LOW/MEDIUM/HIGH)
6. Generates detailed report and annotated video

Usage:
    python scripts/process_live_session.py --session "Datasets/live data/session_20260211_171502"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import argparse

from accelerometer_processor import AccelerometerProcessor, AccelConfig, AccelFeatures
from pothole_detector import YOLODetector, Config


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PotholeEvent:
    """Detected pothole event with fusion data"""
    event_id: int
    timestamp_ms: float
    frame_number: int
    frame_timestamp_ms: float
    
    # Accelerometer data
    accel_peak_g: float
    accel_rms_g: float
    accel_severity: str
    
    # Vision data
    has_vision_detection: bool
    vision_confidence: float = 0.0
    vision_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    vision_area_ratio: float = 0.0
    
    # Fusion result
    fusion_severity: str = "UNKNOWN"
    fusion_score: float = 0.0
    
    # Location (if GPS available)
    latitude: float = 0.0
    longitude: float = 0.0


# =============================================================================
# SESSION PROCESSOR
# =============================================================================

class LiveSessionProcessor:
    """Process live session data with video + accelerometer fusion"""
    
    def __init__(self, session_path: str, yolo_model_path: str = "yolov8n.pt"):
        self.session_path = Path(session_path)
        self.yolo_model_path = yolo_model_path
        
        # Load metadata
        with open(self.session_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize components
        self.accel_processor = None
        self.yolo_detector = None
        self.video_cap = None
        self.frame_timestamps = []
        
        # Results
        self.pothole_events: List[PotholeEvent] = []
        
        print("=" * 70)
        print("  LIVE SESSION PROCESSOR - Video + Accelerometer Fusion")
        print("=" * 70)
        print(f"Session: {self.session_path.name}")
        print(f"Duration: {self.metadata['duration_seconds']:.1f} seconds")
        print(f"Accel samples: {self.metadata['total_accel_samples']}")
        print()
    
    def initialize(self):
        """Initialize all components"""
        
        # 1. Initialize accelerometer processor
        print("[1/4] Initializing accelerometer processor...")
        accel_config = AccelConfig()
        self.accel_processor = AccelerometerProcessor(accel_config)
        accel_csv = self.session_path / "accelerometer" / "accel_log.csv"
        self.accel_processor.initialize(csv_path=str(accel_csv))
        print(f"  ✅ Loaded accelerometer data")
        
        # 2. Load frame timestamps
        print("[2/4] Loading video frame timestamps...")
        frame_ts_file = self.session_path / "video" / "frame_timestamps.txt"
        with open(frame_ts_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            self.frame_timestamps = [float(line.strip()) for line in lines]
        print(f"  ✅ Loaded {len(self.frame_timestamps)} frame timestamps")
        
        # 3. Open video
        print("[3/4] Opening video file...")
        video_file = self.session_path / "video" / "capture.mp4"
        self.video_cap = cv2.VideoCapture(str(video_file))
        if not self.video_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_file}")
        
        total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        print(f"  ✅ Video: {total_frames} frames @ {fps:.1f} FPS")
        
        # 4. Initialize YOLO detector
        print("[4/4] Initializing YOLO detector...")
        self.yolo_detector = YOLODetector(self.yolo_model_path, confidence=0.25)
        if not self.yolo_detector.initialize():
            raise RuntimeError("Failed to initialize YOLO detector")
        print(f"  ✅ YOLO model loaded")
        print()
    
    def detect_pothole_events_from_accel(self, threshold_g: float = 0.3) -> List[Dict]:
        """
        Detect pothole events from accelerometer data
        
        Returns list of events with timestamp and accel features
        """
        print("=" * 70)
        print("  STEP 1: Detecting Pothole Events from Accelerometer")
        print("=" * 70)
        
        # Load accelerometer data
        accel_csv = self.session_path / "accelerometer" / "accel_log.csv"
        accel_df = pd.read_csv(accel_csv)
        
        print(f"Analyzing {len(accel_df)} accelerometer samples...")
        print(f"Detection threshold: {threshold_g}G")
        print()
        
        events = []
        window_size = 50  # 0.5 seconds at 100Hz
        
        # Scan through data with sliding window
        for i in range(0, len(accel_df) - window_size, window_size // 2):
            window = accel_df.iloc[i:i+window_size]
            
            # Calculate magnitude
            ay = window['ay'].values
            az = window['az'].values
            magnitude = np.sqrt(ay**2 + az**2)
            
            peak = np.max(np.abs(magnitude))
            rms = np.sqrt(np.mean(magnitude**2))
            
            # Detect if this is a pothole impact
            if peak > threshold_g or rms > (threshold_g * 0.5):
                timestamp_ms = window['timestamp_ms'].iloc[window_size // 2]
                
                # Classify severity from accelerometer alone
                if peak >= 1.5 or rms >= 0.5:
                    accel_severity = "HIGH"
                elif peak >= 0.5 or rms >= 0.15:
                    accel_severity = "MEDIUM"
                else:
                    accel_severity = "LOW"
                
                events.append({
                    'timestamp_ms': timestamp_ms,
                    'accel_peak_g': float(peak),
                    'accel_rms_g': float(rms),
                    'accel_severity': accel_severity,
                })
        
        # Remove duplicate events (within 1 second)
        filtered_events = []
        last_time = -1000
        for event in events:
            if event['timestamp_ms'] - last_time > 1000:  # 1 second gap
                filtered_events.append(event)
                last_time = event['timestamp_ms']
        
        print(f"✅ Detected {len(filtered_events)} pothole events")
        print()
        
        # Show summary
        severities = [e['accel_severity'] for e in filtered_events]
        from collections import Counter
        counts = Counter(severities)
        print("Accelerometer-based severity distribution:")
        for sev in ['HIGH', 'MEDIUM', 'LOW']:
            if sev in counts:
                print(f"  {sev}: {counts[sev]}")
        print()
        
        return filtered_events
    
    def find_closest_frame(self, timestamp_ms: float) -> Tuple[int, float]:
        """Find video frame closest to given timestamp"""
        
        # Frame timestamps are in milliseconds
        differences = [abs(ft - timestamp_ms) for ft in self.frame_timestamps]
        min_idx = np.argmin(differences)
        
        return min_idx, self.frame_timestamps[min_idx]
    
    def extract_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Extract specific frame from video"""
        
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_cap.read()
        
        if ret:
            return frame
        return None
    
    def process_events_with_vision(self, accel_events: List[Dict]):
        """Process each accelerometer event with vision detection"""
        
        print("=" * 70)
        print("  STEP 2: Processing Events with Vision Detection")
        print("=" * 70)
        print()
        
        for idx, accel_event in enumerate(accel_events, 1):
            timestamp_ms = accel_event['timestamp_ms']
            
            # Find corresponding frame
            frame_num, frame_time = self.find_closest_frame(timestamp_ms)
            time_diff = abs(frame_time - timestamp_ms)
            
            print(f"[{idx}/{len(accel_events)}] Event at {timestamp_ms:.0f}ms")
            print(f"  → Frame #{frame_num} at {frame_time:.0f}ms (Δ{time_diff:.0f}ms)")
            
            # Extract frame
            frame = self.extract_frame(frame_num)
            if frame is None:
                print(f"  ❌ Failed to extract frame")
                continue
            
            # Run YOLO detection
            detections = self.yolo_detector.detect(frame)
            
            # Create pothole event
            event = PotholeEvent(
                event_id=int(idx),
                timestamp_ms=float(timestamp_ms),
                frame_number=int(frame_num),
                frame_timestamp_ms=float(frame_time),
                accel_peak_g=float(accel_event['accel_peak_g']),
                accel_rms_g=float(accel_event['accel_rms_g']),
                accel_severity=accel_event['accel_severity'],
                has_vision_detection=len(detections) > 0,
            )
            
            if detections:
                # Use first (highest confidence) detection
                det = detections[0]
                event.vision_confidence = det['confidence']
                event.vision_bbox = det['bbox']
                
                # Calculate area ratio
                x1, y1, x2, y2 = det['bbox']
                bbox_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                event.vision_area_ratio = bbox_area / frame_area
                
                print(f"  ✅ Vision: Confidence={det['confidence']:.2f}, Area={event.vision_area_ratio:.3f}")
            else:
                print(f"  ⚠️ Vision: No pothole detected")
            
            # Apply fusion classification
            event.fusion_severity, event.fusion_score = self._classify_with_fusion(event)
            
            print(f"  🎯 Fusion Result: {event.fusion_severity} (score={event.fusion_score:.2f})")
            print()
            
            self.pothole_events.append(event)
    
    def _classify_with_fusion(self, event: PotholeEvent) -> Tuple[str, float]:
        """Apply vision + accelerometer fusion to classify severity"""
        
        vision_weight = 0.6
        accel_weight = 0.4
        
        # Vision score (0-1)
        if event.has_vision_detection:
            vision_score = event.vision_confidence * min(event.vision_area_ratio / 0.15, 1.0)
        else:
            vision_score = 0.0
        
        # Accelerometer score (0-1)
        accel_score = min(event.accel_peak_g / 3.0, 1.0)
        
        # Weighted fusion
        if event.has_vision_detection:
            fusion_score = vision_weight * vision_score + accel_weight * accel_score
        else:
            # No vision, rely more on accelerometer
            fusion_score = accel_score
        
        # Classification rules
        if event.accel_peak_g >= 2.0:
            # Strong impact overrides
            severity = "HIGH"
        elif event.has_vision_detection and event.vision_confidence > 0.8 and event.vision_area_ratio > 0.1 and event.accel_peak_g > 0.5:
            # Large visual + moderate impact
            severity = "HIGH"
        elif fusion_score > 0.7:
            severity = "HIGH"
        elif fusion_score > 0.4:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        return severity, fusion_score
    
    def generate_report(self, output_path: str = None):
        """Generate detailed report of results"""
        
        if output_path is None:
            output_path = self.session_path / "pothole_detection_report.json"
        
        print("=" * 70)
        print("  STEP 3: Generating Report")
        print("=" * 70)
        print()
        
        # Summary statistics
        total_events = len(self.pothole_events)
        vision_detected = sum(1 for e in self.pothole_events if e.has_vision_detection)
        
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for event in self.pothole_events:
            severity_counts[event.fusion_severity] += 1
        
        report = {
            'session_info': {
                'session_id': self.metadata['session_id'],
                'start_time': self.metadata['start_time'],
                'duration_seconds': self.metadata['duration_seconds'],
            },
            'detection_summary': {
                'total_pothole_events': total_events,
                'vision_detected': vision_detected,
                'vision_detection_rate': f"{(vision_detected/total_events*100):.1f}%" if total_events > 0 else "0%",
                'severity_distribution': severity_counts,
            },
            'events': [asdict(event) for event in self.pothole_events],
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved: {output_path}")
        print()
        print("DETECTION SUMMARY:")
        print(f"  Total pothole events: {total_events}")
        print(f"  Vision detected: {vision_detected} ({vision_detected/total_events*100:.1f}%)" if total_events > 0 else "  Vision detected: 0")
        print(f"  Severity distribution:")
        for sev in ['HIGH', 'MEDIUM', 'LOW']:
            count = severity_counts[sev]
            pct = (count / total_events * 100) if total_events > 0 else 0
            print(f"    {sev}: {count} ({pct:.1f}%)")
        print()
        
        return report
    
    def create_annotated_video(self, output_path: str = None, max_events: int = 20, 
                              last_n_minutes: float = 2.0, vision_only: bool = True):
        """Create video clips of detected potholes with annotations matching test results style"""
        
        if output_path is None:
            output_path = self.session_path / "annotated_potholes.mp4"
        
        print("=" * 70)
        print("  STEP 4: Creating Annotated Video Clips")
        print("=" * 70)
        print()
        
        # Filter events
        filtered_events = self.pothole_events
        
        # Filter for vision-detected events only if requested
        if vision_only:
            filtered_events = [e for e in filtered_events if e.has_vision_detection]
            print(f"Filtering for vision-detected events only: {len(filtered_events)} found")
        
        # Filter for last N minutes if specified
        if last_n_minutes > 0:
            session_duration = self.metadata['duration_seconds']
            cutoff_time_ms = (session_duration - last_n_minutes * 60) * 1000
            filtered_events = [e for e in filtered_events if e.timestamp_ms >= cutoff_time_ms]
            print(f"Filtering for last {last_n_minutes} minutes: {len(filtered_events)} events")
        
        if not filtered_events:
            print("❌ No events found matching criteria!")
            print("   Try: --no-vision-only to include all events")
            return
        
        # Limit to max events
        events_to_process = filtered_events[:max_events]
        
        print(f"Creating clips for {len(events_to_process)} events...")
        print("Style: Blue boxes with 'Potholes' labels (matching test results)")
        print()
        
        # Show event details
        print("Events to be annotated:")
        for i, event in enumerate(events_to_process[:10], 1):
            time_sec = event.timestamp_ms / 1000
            print(f"  {i}. Time: {time_sec:.1f}s | Conf: {event.vision_confidence:.2f} | "
                  f"Severity: {event.fusion_severity} | Accel: {event.accel_peak_g:.2f}G")
        if len(events_to_process) > 10:
            print(f"  ... and {len(events_to_process) - 10} more")
        print()
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Original FPS for smooth playback
        frame_size = (1280, 960)  # From metadata
        
        out = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        
        frames_written = 0
        
        for event in events_to_process:
            # Extract frames around event
            # Before: 1 second (30 frames)
            # During: Show detection for 2 seconds (60 frames) 
            # After: 1 second (30 frames)
            
            start_frame = max(0, event.frame_number - 30)
            detection_frame = event.frame_number
            end_frame = min(len(self.frame_timestamps) - 1, event.frame_number + 90)
            
            for frame_num in range(start_frame, end_frame + 1):
                frame = self.extract_frame(frame_num)
                if frame is None:
                    continue
                
                # Annotate frame
                annotated = frame.copy()
                
                # Determine if we should show detection
                show_detection = (frame_num >= detection_frame and 
                                frame_num < detection_frame + 60)  # Show for 2 seconds
                
                if show_detection and event.has_vision_detection:
                    x1, y1, x2, y2 = event.vision_bbox
                    
                    # Use BLUE color to match test results style
                    color = (255, 0, 0)  # Blue in BGR
                    thickness = 3
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                    
                    # Create label matching test results: "Potholes 0.XX"
                    label = f"Potholes {event.vision_confidence:.2f}"
                    
                    # Calculate label size for background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    font_thickness = 2
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, font, font_scale, font_thickness
                    )
                    
                    # Draw label background (blue)
                    label_y = y1 - 10 if y1 > 30 else y1 + label_height + 10
                    cv2.rectangle(
                        annotated,
                        (x1, label_y - label_height - 5),
                        (x1 + label_width + 10, label_y + baseline),
                        color,
                        -1  # Filled
                    )
                    
                    # Draw label text (white)
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 5, label_y - 5),
                        font,
                        font_scale,
                        (255, 255, 255),  # White text
                        font_thickness,
                        cv2.LINE_AA
                    )
                
                # Add event info overlay at top
                if show_detection:
                    # Create semi-transparent overlay bar
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (0, 0), (1280, 60), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
                    
                    # Event info
                    info_text = f"Event #{event.event_id} | {event.fusion_severity} Severity"
                    cv2.putText(annotated, info_text, (20, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Sensor data
                    sensor_text = f"Accel: {event.accel_peak_g:.2f}G | Fusion Score: {event.fusion_score:.2f}"
                    cv2.putText(annotated, sensor_text, (20, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                out.write(annotated)
                frames_written += 1
        
        out.release()
        
        print(f"✅ Annotated video saved: {output_path}")
        print(f"   Total frames written: {frames_written}")
        print(f"   Duration: ~{frames_written / fps:.1f} seconds")
        print(f"   Each detection shown for 2 seconds")
        print()
    
    def cleanup(self):
        """Release resources"""
        if self.video_cap:
            self.video_cap.release()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Process live session with video + accelerometer fusion')
    parser.add_argument('--session', type=str, 
                       default='Datasets/live data/session_20260211_171502',
                       help='Path to session directory')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Accelerometer detection threshold (G)')
    parser.add_argument('--max-clips', type=int, default=20,
                       help='Maximum number of video clips to generate')
    parser.add_argument('--last-minutes', type=float, default=2.0,
                       help='Process only last N minutes of video (0 = all)')
    parser.add_argument('--no-vision-only', action='store_true',
                       help='Include events without vision detection')
    
    args = parser.parse_args()
    
    # Create processor
    processor = LiveSessionProcessor(args.session, args.model)
    
    try:
        # Initialize
        processor.initialize()
        
        # Step 1: Detect events from accelerometer
        accel_events = processor.detect_pothole_events_from_accel(threshold_g=args.threshold)
        
        # Step 2: Process with vision
        processor.process_events_with_vision(accel_events)
        
        # Step 3: Generate report
        processor.generate_report()
        
        # Step 4: Create annotated video
        processor.create_annotated_video(
            max_events=args.max_clips,
            last_n_minutes=args.last_minutes,
            vision_only=not args.no_vision_only
        )
        
        print("=" * 70)
        print("  ✅ PROCESSING COMPLETE!")
        print("=" * 70)
        
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main()

