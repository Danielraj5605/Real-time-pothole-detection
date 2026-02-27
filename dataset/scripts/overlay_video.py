#!/usr/bin/env python3
"""
Video Overlay Tool
Overlays accelerometer data on video frames to visualize sensor readings.

RUN THIS ON YOUR LAPTOP (requires ffmpeg and opencv-python)

Usage:
    python overlay_video.py <session_folder> [options]

Options:
    --offset MILLISECONDS   Offset to apply to accelerometer timestamps (default: auto-detect)
                           Positive = accelerometer data is ahead of video
                           Negative = accelerometer data is behind video

Example:
    python overlay_video.py ./session_20260201_005615
    python overlay_video.py ./session_20260201_005615 --offset 5000
"""

import os
import sys
import csv
import subprocess
import argparse
from pathlib import Path
import json

# Check for required packages
try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: opencv-python and numpy required.")
    print("Install with: pip install opencv-python numpy")
    sys.exit(1)


def load_accel_data(csv_path):
    """Load accelerometer data from CSV file."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'timestamp_ms': int(row['timestamp_ms']),
                'ax': float(row['ax']),
                'ay': float(row['ay']),
                'az': float(row['az']),
                'gx': float(row['gx']),
                'gy': float(row['gy']),
                'gz': float(row['gz'])
            })
    return data


def load_labels(json_path):
    """Load pothole labels from JSON file."""
    if not json_path.exists():
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data.get('events', [])


def find_closest_accel(accel_data, timestamp_ms):
    """Find the closest accelerometer reading to a given timestamp."""
    if not accel_data:
        return None
    
    # Binary search would be faster, but linear is fine for this
    closest = min(accel_data, key=lambda x: abs(x['timestamp_ms'] - timestamp_ms))
    return closest


def convert_video_to_mp4(input_path, mp4_path):
    """Convert H.264 or MJPEG to MP4 if needed."""
    if mp4_path.exists():
        return True
    
    print(f"Converting {input_path} to MP4...")
    
    # Find ffmpeg
    ffmpeg_cmd = "ffmpeg"
    # Try winget path on Windows
    winget_ffmpeg = Path(os.environ.get('LOCALAPPDATA', '')) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    if winget_ffmpeg.exists():
        ffmpeg_cmd = str(winget_ffmpeg)
    
    # Different commands for different formats
    if str(input_path).endswith('.mjpeg'):
        cmd = [
            ffmpeg_cmd, '-y',
            '-framerate', '15',  # MJPEG recorded at 15fps
            '-i', str(input_path),
            '-c:v', 'libx264',  # Re-encode MJPEG to H.264
            '-preset', 'fast',
            '-crf', '23',
            str(mp4_path)
        ]
    else:
        # H.264 - just copy
        cmd = [
            ffmpeg_cmd, '-y',
            '-framerate', '30',
            '-i', str(input_path),
            '-c', 'copy',
            str(mp4_path)
        ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return mp4_path.exists()
    except Exception as e:
        print(f"Error converting video: {e}")
        return False


def draw_overlay(frame, accel, magnitude, frame_time_ms, labels, frame_num):
    """Draw accelerometer overlay on video frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Colors
    white = (255, 255, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    
    # Choose magnitude color based on value
    if magnitude > 2.0:
        mag_color = red
    elif magnitude > 1.5:
        mag_color = yellow
    else:
        mag_color = green
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 35
    
    cv2.putText(frame, f"Time: {frame_time_ms/1000:.2f}s  Frame: {frame_num}", 
                (20, y_offset), font, 0.5, white, 1)
    y_offset += 25
    
    cv2.putText(frame, f"Accel X: {accel['ax']:+.3f} g", 
                (20, y_offset), font, 0.6, white, 1)
    y_offset += 22
    
    cv2.putText(frame, f"Accel Y: {accel['ay']:+.3f} g", 
                (20, y_offset), font, 0.6, white, 1)
    y_offset += 22
    
    cv2.putText(frame, f"Accel Z: {accel['az']:+.3f} g", 
                (20, y_offset), font, 0.6, white, 1)
    y_offset += 25
    
    cv2.putText(frame, f"Magnitude: {magnitude:.3f} g", 
                (20, y_offset), font, 0.7, mag_color, 2)
    
    # Draw magnitude bar
    bar_x = 200
    bar_y = 40
    bar_width = 140
    bar_height = 100
    
    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (50, 50, 50), -1)
    
    # Magnitude fill (clamp to 3g max for display)
    fill_height = int(min(magnitude / 3.0, 1.0) * bar_height)
    fill_y = bar_y + bar_height - fill_height
    cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_width, bar_y + bar_height), 
                  mag_color, -1)
    
    # Reference lines
    cv2.line(frame, (bar_x, bar_y + int(bar_height * 2/3)), 
             (bar_x + bar_width, bar_y + int(bar_height * 2/3)), white, 1)  # 1g line
    cv2.putText(frame, "1g", (bar_x + bar_width + 5, bar_y + int(bar_height * 2/3) + 5), 
                font, 0.4, white, 1)
    
    cv2.line(frame, (bar_x, bar_y + int(bar_height * 1/3)), 
             (bar_x + bar_width, bar_y + int(bar_height * 1/3)), yellow, 1)  # 2g line
    cv2.putText(frame, "2g", (bar_x + bar_width + 5, bar_y + int(bar_height * 1/3) + 5), 
                font, 0.4, yellow, 1)
    
    # Check if we're near a label
    for label in labels:
        label_time = label.get('timestamp_ms', 0)
        if abs(frame_time_ms - label_time) < 500:  # Within 500ms of label
            # Flash red border
            cv2.rectangle(frame, (0, 0), (w-1, h-1), red, 8)
            cv2.putText(frame, "POTHOLE LABELED!", (w//2 - 100, 50), 
                        font, 1.0, red, 3)
            break
    
    return frame


def create_overlay_video(session_dir, output_path=None, offset_ms=None):
    """Create video with accelerometer overlay.
    
    Args:
        session_dir: Path to session directory
        output_path: Output video path (optional)
        offset_ms: Offset in milliseconds to apply to accelerometer data.
                  Positive = accel data is ahead, so we shift forward.
                  If None, auto-detect based on duration mismatch.
    """
    session_dir = Path(session_dir)
    
    # Paths - check for MJPEG first (new format), then H.264 (old format)
    mjpeg_path = session_dir / "video" / "raw_capture.mjpeg"
    h264_path = session_dir / "video" / "raw_capture.h264"
    mp4_path = session_dir / "video" / "capture.mp4"
    accel_path = session_dir / "accelerometer" / "accel_log.csv"
    labels_path = session_dir / "labels" / "events.json"
    metadata_path = session_dir / "metadata.json"
    
    if output_path is None:
        output_path = session_dir / "video" / "capture_overlay.mp4"
    else:
        output_path = Path(output_path)
    
    # Check files exist
    if not accel_path.exists():
        print(f"ERROR: Accelerometer data not found: {accel_path}")
        return False
    
    # Convert video if needed
    if not mp4_path.exists():
        if mjpeg_path.exists():
            if not convert_video_to_mp4(mjpeg_path, mp4_path):
                print("ERROR: Could not convert MJPEG video")
                return False
        elif h264_path.exists():
            if not convert_video_to_mp4(h264_path, mp4_path):
                print("ERROR: Could not convert H.264 video")
                return False
        else:
            print(f"ERROR: No video file found (checked MJPEG and H.264)")
            return False
    
    # Load data
    print("Loading accelerometer data...")
    accel_data = load_accel_data(accel_path)
    print(f"  Loaded {len(accel_data)} accelerometer samples")
    
    labels = load_labels(labels_path)
    print(f"  Loaded {len(labels)} pothole labels")
    
    # Open video
    print(f"Processing video: {mp4_path}")
    cap = cv2.VideoCapture(str(mp4_path))
    
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_ms = int((total_frames / fps) * 1000)
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Video duration: {video_duration_ms/1000:.1f}s")
    
    # Get accelerometer duration
    if accel_data:
        accel_duration_ms = accel_data[-1]['timestamp_ms'] - accel_data[0]['timestamp_ms']
        print(f"  Accel duration: {accel_duration_ms/1000:.1f}s")
    else:
        accel_duration_ms = 0
    
    # Load PTS file for precise frame timestamps (milliseconds since first frame)
    pts_path = session_dir / "video" / "frame_timestamps.txt"
    frame_pts = []
    if pts_path.exists():
        try:
            with open(pts_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Handle both integer (microseconds) and float (milliseconds) formats
                    if line and not line.startswith('#'):
                        try:
                            val = float(line)
                            # If value > 1000000, assume microseconds; convert to ms
                            if val > 1000000:
                                frame_pts.append(val / 1000)  # microseconds -> ms
                            else:
                                frame_pts.append(val)  # already in ms
                        except ValueError:
                            continue
            print(f"  Loaded {len(frame_pts)} frame PTS timestamps")
        except Exception as e:
            print(f"  Warning: Could not load PTS file: {e}")
    
    # ================================================================
    # UNIFIED TIMESTAMP SYNC (v2)
    # ================================================================
    # New metadata fields from collect_data.py:
    # - T0_ns: perf_counter_ns at video process start
    # - camera_start_ns: estimated time from T0 to first frame capture (~150ms)
    # - accel_start_ns: time from T0 when accelerometer sampling started
    #
    # Sync formula:
    #   frame_absolute_ms = camera_start_ms + frame_pts[i]
    #   accel timestamps in CSV are relative to T0
    #   So: find accel sample where accel.timestamp_ms ≈ frame_absolute_ms
    # ================================================================
    
    # Load metadata
    metadata = {}
    camera_start_ms = 150  # Default: ~150ms camera warmup
    accel_start_ms = 3000  # Default: ~3s stabilization wait
    use_unified_sync = False
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
                # Check for unified timestamp fields (v2)
                if 'camera_start_ns' in metadata and 'accel_start_ns' in metadata:
                    camera_start_ms = metadata['camera_start_ns'] / 1_000_000
                    accel_start_ms = metadata['accel_start_ns'] / 1_000_000
                    use_unified_sync = True
                    print(f"  Using unified timestamp sync (v2)")
                    print(f"    Camera warmup estimate: T0 + {camera_start_ms:.0f}ms (not used)")
                    print(f"    Accel start: T0 + {accel_start_ms:.0f}ms")
                else:
                    # Legacy: use epoch-based sync
                    video_process_start = metadata.get('video_process_start_epoch')
                    accel_start_epoch = metadata.get('video_recording_start_epoch')
                    
                    if video_process_start and accel_start_epoch:
                        offset_seconds = accel_start_epoch - video_process_start
                        accel_start_ms = offset_seconds * 1000
                        print(f"  Using legacy epoch-based sync")
                        print(f"  Accel started {offset_seconds:.2f}s after video process")
        except Exception as e:
            print(f"  Warning: Could not load metadata: {e}")
    
    # ================================================================
    # SYNC CALCULATION FIX:
    # ================================================================
    # The video PTS=0 and accel samples BOTH start after the 3-second
    # stabilization period. They happen at approximately the same time!
    # 
    # Therefore:
    #   frame at PTS=0 corresponds to accel at first_accel_timestamp
    #   frame at PTS=P corresponds to accel at first_accel_timestamp + P
    #
    # We use the first accel sample timestamp as the base, not camera_start_ms.
    # ================================================================
    first_accel_ts = accel_data[0]['timestamp_ms'] if accel_data else accel_start_ms
    
    if offset_ms is None:
        # CALIBRATED OFFSET: +1700ms compensates for timing difference between
        # video PTS and accelerometer timestamps. This value was determined by
        # manual testing: shaking the camera and adjusting until overlay matches
        # the visual motion in the video.
        offset_ms = 1700
        print(f"  Sync: frame at PTS=P maps to accel at {first_accel_ts:.0f}ms + P + {offset_ms}ms")
    else:
        print(f"  Using manual offset: {offset_ms}ms")
    
    # Note: With the corrected sync formula (first_accel_ts + PTS), 
    # video PTS=0 naturally aligns with first accel sample.
    # No video trimming is needed.
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("ERROR: Could not create output video")
        return False
    
    print(f"\nCreating overlay video: {output_path}")
    
    # Track frame number (adjusted for skip)
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    output_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ============================================================
        # UNIFIED TIMESTAMP: Calculate frame absolute time (relative to T0)
        # ============================================================
        if frame_num < len(frame_pts):
            # Use actual PTS timestamp
            frame_pts_ms = frame_pts[frame_num]
        else:
            # Fallback: calculate from FPS
            frame_pts_ms = (frame_num / fps) * 1000
        
        # SIMPLE SYNC: Use PTS + first_accel_ts as the lookup time
        # This assumes video PTS=0 corresponds to when accel started
        # The offset_ms allows manual adjustment
        frame_absolute_ms = first_accel_ts + frame_pts_ms + offset_ms
        
        # Find accelerometer sample at this absolute time (also relative to T0)
        accel = find_closest_accel(accel_data, frame_absolute_ms)
        
        # Debug output for first 5 frames (after skipping)
        if output_frame_count < 5:
            accel_ts = accel['timestamp_ms'] if accel else 'N/A'
            print(f"  DEBUG: frame_num={frame_num}, PTS={frame_pts_ms:.0f}ms, absolute={frame_absolute_ms:.0f}ms, accel_ts={accel_ts}ms")
        
        if accel:
            # Calculate magnitude
            magnitude = (accel['ax']**2 + accel['ay']**2 + accel['az']**2) ** 0.5
            
            # Draw overlay (show PTS time for display, not absolute time)
            frame = draw_overlay(frame, accel, magnitude, int(frame_pts_ms), labels, frame_num)
        
        # Write frame
        out.write(frame)
        
        frame_num += 1
        output_frame_count += 1
        
        # Progress update
        if frame_num % 100 == 0:
            print(f"  Processed {frame_num}/{total_frames} frames ({100*frame_num/total_frames:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\nDone! Output saved to: {output_path}")
    print(f"Total frames processed: {frame_num}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Overlay accelerometer data on video",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('session_dir', type=str,
                        help='Path to session directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output video path (default: session_dir/video/capture_overlay.mp4)')
    parser.add_argument('--offset', type=int, default=None,
                        help='Offset in ms to apply to accelerometer data (default: auto-detect). '
                             'Positive = accel ahead of video')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 50)
    print("VIDEO OVERLAY TOOL")
    print("=" * 50 + "\n")
    
    success = create_overlay_video(args.session_dir, args.output, args.offset)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
