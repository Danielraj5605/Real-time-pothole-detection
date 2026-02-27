#!/usr/bin/env python3
"""
Frame Extraction Tool
Extract frames from H.264 video and align with sensor data.

RUN THIS ON YOUR LAPTOP (not on the Pi) after copying the data.

Usage:
    python extract_frames.py <session_folder> [options]

Examples:
    python extract_frames.py ./session_20260131_143000
    python extract_frames.py ./session_20260131_143000 --interval 100 --around-labels
"""

import os
import sys
import json
import csv
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def convert_h264_to_mp4(input_path, output_path):
    """
    Convert H.264 raw video to MP4 using ffmpeg.
    
    Args:
        input_path: Path to .h264 file
        output_path: Path for output .mp4 file
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Converting video: {input_path} -> {output_path}")
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-framerate", "30",
        "-i", str(input_path),
        "-c", "copy",
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return False
        print("Conversion complete!")
        return True
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Please install ffmpeg first.")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def extract_frames(video_path, output_dir, interval_ms=100, fps=30):
    """
    Extract frames from video at specified intervals.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        interval_ms: Interval between frames in milliseconds
        fps: Video framerate
        
    Returns:
        List of extracted frame info (timestamp, path)
    """
    print(f"Extracting frames every {interval_ms}ms...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate frame interval
    frame_interval = int((interval_ms / 1000) * fps)
    if frame_interval < 1:
        frame_interval = 1
    
    # Use ffmpeg to extract frames
    # Output pattern: frame_NNNNNN.jpg
    output_pattern = str(output_dir / "frame_%06d.jpg")
    
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"select='not(mod(n,{frame_interval}))'",
        "-vsync", "vfr",
        "-q:v", "2",  # High quality JPEG
        output_pattern
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Check if it's just a warning
            if "frame=" not in result.stderr:
                print(f"ffmpeg warning: {result.stderr[-500:]}")
    except Exception as e:
        print(f"ERROR extracting frames: {e}")
        return []
    
    # List extracted frames
    frames = sorted(output_dir.glob("frame_*.jpg"))
    print(f"Extracted {len(frames)} frames")
    
    # Generate frame info with estimated timestamps
    frame_info = []
    for i, frame_path in enumerate(frames):
        timestamp_ms = i * interval_ms
        frame_info.append({
            "frame_number": i,
            "timestamp_ms": timestamp_ms,
            "path": str(frame_path)
        })
    
    return frame_info


def extract_frames_around_labels(video_path, output_dir, labels, window_ms=500, fps=30):
    """
    Extract frames around label events.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        labels: List of label events with timestamp_ms
        window_ms: Time window around each label (before and after)
        fps: Video framerate
        
    Returns:
        List of extracted frame info
    """
    if not labels:
        print("No labels found, skipping label-based extraction")
        return []
    
    print(f"Extracting frames around {len(labels)} labels (±{window_ms}ms window)...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_info = []
    
    for i, label in enumerate(labels):
        label_time_ms = label.get("timestamp_ms", 0)
        
        # Calculate frame range
        start_time_s = max(0, (label_time_ms - window_ms) / 1000)
        duration_s = (window_ms * 2) / 1000
        
        # Output pattern for this label
        label_dir = output_dir / f"label_{i:04d}_{label_time_ms}"
        label_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = str(label_dir / "frame_%04d.jpg")
        
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time_s),
            "-i", str(video_path),
            "-t", str(duration_s),
            "-q:v", "2",
            output_pattern
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True)
        except Exception as e:
            print(f"  Warning: Failed to extract frames for label {i}: {e}")
            continue
        
        # List extracted frames for this label
        frames = sorted(label_dir.glob("frame_*.jpg"))
        
        for j, frame_path in enumerate(frames):
            frame_time = start_time_s * 1000 + (j * (1000 / fps))
            frame_info.append({
                "label_index": i,
                "label_timestamp_ms": label_time_ms,
                "frame_timestamp_ms": int(frame_time),
                "path": str(frame_path)
            })
        
        print(f"  Label {i}: extracted {len(frames)} frames at {label_time_ms}ms")
    
    print(f"Total: {len(frame_info)} frames extracted around labels")
    return frame_info


def match_sensor_data(frame_info, accel_path, gps_path):
    """
    Match frames with closest sensor readings.
    
    Args:
        frame_info: List of frame info dicts with timestamp_ms
        accel_path: Path to accelerometer CSV
        gps_path: Path to GPS CSV
        
    Returns:
        Updated frame_info with matched sensor data
    """
    print("Matching sensor data to frames...")
    
    # Load accelerometer data
    accel_data = []
    if Path(accel_path).exists():
        with open(accel_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                accel_data.append({
                    'timestamp_ms': int(row['timestamp_ms']),
                    'ax': float(row['ax']),
                    'ay': float(row['ay']),
                    'az': float(row['az']),
                    'gx': float(row['gx']),
                    'gy': float(row['gy']),
                    'gz': float(row['gz'])
                })
    
    # Load GPS data
    gps_data = []
    if Path(gps_path).exists():
        with open(gps_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gps_data.append({
                    'timestamp_ms': int(row['timestamp_ms']),
                    'latitude': float(row['latitude']) if row['latitude'] else None,
                    'longitude': float(row['longitude']) if row['longitude'] else None,
                    'speed_kmh': float(row['speed_kmh']) if row['speed_kmh'] else 0
                })
    
    # Match each frame to closest sensor readings
    for frame in frame_info:
        frame_ts = frame.get('timestamp_ms', 0)
        
        # Find closest accelerometer reading
        if accel_data:
            closest_accel = min(accel_data, key=lambda x: abs(x['timestamp_ms'] - frame_ts))
            frame['accel'] = {
                'ax': closest_accel['ax'],
                'ay': closest_accel['ay'],
                'az': closest_accel['az'],
                'magnitude': (closest_accel['ax']**2 + closest_accel['ay']**2 + closest_accel['az']**2)**0.5
            }
        
        # Find closest GPS reading
        if gps_data:
            closest_gps = min(gps_data, key=lambda x: abs(x['timestamp_ms'] - frame_ts))
            frame['gps'] = {
                'latitude': closest_gps['latitude'],
                'longitude': closest_gps['longitude'],
                'speed_kmh': closest_gps['speed_kmh']
            }
    
    print(f"Matched sensor data to {len(frame_info)} frames")
    return frame_info


def generate_annotation_template(frame_info, output_path, labels=None):
    """
    Generate annotation template file for labeling.
    
    Args:
        frame_info: List of frame info dicts
        output_path: Output JSON path
        labels: Original label events (to mark as pothole candidates)
    """
    print(f"Generating annotation template: {output_path}")
    
    # Create label timestamp set for quick lookup
    label_timestamps = set()
    if labels:
        for label in labels:
            label_timestamps.add(label.get('timestamp_ms', 0))
    
    annotations = []
    for frame in frame_info:
        frame_ts = frame.get('timestamp_ms', 0)
        
        # Check if this frame is near a label (±500ms)
        is_labeled = any(abs(frame_ts - lt) < 500 for lt in label_timestamps)
        
        annotation = {
            "image_path": frame.get('path', ''),
            "timestamp_ms": frame_ts,
            "is_pothole_candidate": is_labeled,
            "annotations": [],  # To be filled during manual labeling
            "sensor_data": {
                "accel": frame.get('accel', {}),
                "gps": frame.get('gps', {})
            }
        }
        annotations.append(annotation)
    
    with open(output_path, 'w') as f:
        json.dump({"frames": annotations}, f, indent=2)
    
    pothole_candidates = sum(1 for a in annotations if a['is_pothole_candidate'])
    print(f"Generated template with {len(annotations)} frames ({pothole_candidates} pothole candidates)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from session video and match with sensor data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('session_dir', type=str,
                        help='Path to session directory')
    parser.add_argument('--interval', type=int, default=100,
                        help='Frame extraction interval in milliseconds (default: 100)')
    parser.add_argument('--around-labels', action='store_true',
                        help='Extract frames only around label events (±500ms)')
    parser.add_argument('--window', type=int, default=500,
                        help='Time window around labels in ms (default: 500)')
    parser.add_argument('--skip-convert', action='store_true',
                        help='Skip H.264 to MP4 conversion')
    
    args = parser.parse_args()
    
    session_dir = Path(args.session_dir)
    
    if not session_dir.exists():
        print(f"ERROR: Session directory not found: {session_dir}")
        sys.exit(1)
    
    print(f"\n{'=' * 50}")
    print(f"FRAME EXTRACTION TOOL")
    print(f"Session: {session_dir.name}")
    print(f"{'=' * 50}\n")
    
    # Paths
    h264_path = session_dir / "video" / "raw_capture.h264"
    mp4_path = session_dir / "video" / "capture.mp4"
    frames_dir = session_dir / "frames"
    labels_path = session_dir / "labels" / "events.json"
    accel_path = session_dir / "accelerometer" / "accel_log.csv"
    gps_path = session_dir / "gps" / "gps_log.csv"
    annotations_path = session_dir / "labels" / "annotations.json"
    
    # Load labels
    labels = []
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            data = json.load(f)
            labels = data.get('events', [])
        print(f"Loaded {len(labels)} label events")
    
    # Convert H.264 to MP4
    video_path = mp4_path
    if h264_path.exists() and not args.skip_convert:
        if not mp4_path.exists() or h264_path.stat().st_mtime > mp4_path.stat().st_mtime:
            if not convert_h264_to_mp4(h264_path, mp4_path):
                print("WARNING: Video conversion failed, trying to use H.264 directly")
                video_path = h264_path
    elif mp4_path.exists():
        print("Using existing MP4 file")
    else:
        print("WARNING: No video file found")
        video_path = None
    
    # Extract frames
    frame_info = []
    if video_path and video_path.exists():
        if args.around_labels and labels:
            frame_info = extract_frames_around_labels(
                video_path, frames_dir / "labels",
                labels, window_ms=args.window
            )
        else:
            frame_info = extract_frames(
                video_path, frames_dir,
                interval_ms=args.interval
            )
    
    # Match sensor data
    if frame_info:
        frame_info = match_sensor_data(frame_info, accel_path, gps_path)
    
    # Generate annotation template
    if frame_info:
        generate_annotation_template(frame_info, annotations_path, labels)
    
    print(f"\n{'=' * 50}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 50}")
    print(f"\nFrames: {len(frame_info)}")
    print(f"Labels: {len(labels)}")
    print(f"\nOutput:")
    print(f"  Video: {video_path}")
    print(f"  Frames: {frames_dir}")
    print(f"  Annotations: {annotations_path}")
    print(f"\nNext steps:")
    print("  1. Review frames and manually annotate bounding boxes")
    print("  2. Use annotations.json as reference for pothole candidates")
    print("  3. Export to YOLO format for training")


if __name__ == "__main__":
    main()
