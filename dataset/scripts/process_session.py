#!/usr/bin/env python3
"""
Process Session - All-in-one post-processing script
Copies session from Pi, converts video, and creates overlay.

Usage:
    python process_session.py <session_name>
    python process_session.py session_20260201_012024
    python process_session.py latest                    # Process most recent session

This script will:
1. Copy session data from Raspberry Pi to local folder
2. Convert H.264 video to MP4
3. Create overlay video with accelerometer data
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Configuration - UPDATE THESE FOR YOUR SETUP
PI_USER = "raspberrypi"
PI_HOST = "10.91.79.190"
PI_DATA_DIR = "~/pothole_dataset/data"
LOCAL_DATA_DIR = Path(__file__).parent.parent / "testing"  # Sessions saved here

# Find ffmpeg
def get_ffmpeg_path():
    """Find ffmpeg executable."""
    # Try PATH first
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if result.returncode == 0:
            return "ffmpeg"
    except FileNotFoundError:
        pass
    
    # Try winget installation path on Windows
    winget_path = Path(os.environ.get('LOCALAPPDATA', '')) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    if winget_path.exists():
        return str(winget_path)
    
    return None


def run_command(cmd, description, check=True):
    """Run a shell command with nice output."""
    print(f"\n→ {description}")
    print(f"  Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
        
        if check and result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return False
        
        print(f"  ✓ Done")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def list_remote_sessions(pi_host):
    """List sessions on the Raspberry Pi."""
    # Use $HOME instead of ~ for proper expansion in SSH
    cmd = f'ssh {PI_USER}@{pi_host} "ls -1 $HOME/pothole_dataset/data/"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        return []
    
    sessions = [s.strip() for s in result.stdout.strip().split('\n') if s.strip().startswith('session_')]
    return sorted(sessions)


def copy_from_pi(session_name, local_dir, pi_host):
    """Copy session from Raspberry Pi."""
    # Use full path instead of tilde for SCP compatibility
    remote_path = f"{PI_USER}@{pi_host}:/home/{PI_USER}/pothole_dataset/data/{session_name}"
    local_path = local_dir / session_name
    
    if local_path.exists():
        print(f"  Session already exists locally: {local_path}")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            return local_path
    
    cmd = ["scp", "-r", remote_path, str(local_dir)]
    if run_command(cmd, f"Copying {session_name} from Pi"):
        return local_path
    return None


def convert_to_mp4(session_dir, ffmpeg_path):
    """Convert MJPEG or H.264 to MP4."""
    mjpeg_path = session_dir / "video" / "raw_capture.mjpeg"
    h264_path = session_dir / "video" / "raw_capture.h264"
    pts_path = session_dir / "video" / "frame_timestamps.txt"
    mp4_path = session_dir / "video" / "capture.mp4"
    
    if mp4_path.exists():
        print(f"  MP4 already exists: {mp4_path}")
        return True
    
    if mjpeg_path.exists():
        # MJPEG needs re-encoding - recorded at 15fps
        cmd = [ffmpeg_path, "-y", "-framerate", "15", "-i", str(mjpeg_path),
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(mp4_path)]
        return run_command(cmd, "Converting MJPEG to MP4")
    elif h264_path.exists():
        # Calculate actual framerate from PTS file if available
        framerate = "30"  # Default fallback
        if pts_path.exists():
            try:
                with open(pts_path, 'r') as f:
                    lines = f.readlines()
                # Skip header line if present
                timestamps = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            timestamps.append(float(line))
                        except ValueError:
                            continue
                
                if len(timestamps) >= 2:
                    # Calculate actual FPS: frames / duration_in_seconds
                    duration_ms = timestamps[-1] - timestamps[0]
                    num_frames = len(timestamps)
                    if duration_ms > 0:
                        actual_fps = (num_frames - 1) / (duration_ms / 1000.0)
                        framerate = f"{actual_fps:.2f}"
                        print(f"  Detected framerate from PTS: {actual_fps:.2f} fps ({num_frames} frames over {duration_ms/1000:.1f}s)")
            except Exception as e:
                print(f"  Warning: Could not read PTS file: {e}, using default 30fps")
        
        # H.264 can be copied directly
        cmd = [ffmpeg_path, "-y", "-framerate", framerate, "-i", str(h264_path), "-c", "copy", str(mp4_path)]
        return run_command(cmd, f"Converting H.264 to MP4 at {framerate}fps")
    else:
        print(f"  No video file found (checked MJPEG and H.264)")
        return False


def create_overlay(session_dir, offset_ms=None):
    """Create overlay video."""
    overlay_script = Path(__file__).parent / "overlay_video.py"
    
    cmd = [sys.executable, str(overlay_script), str(session_dir)]
    if offset_ms is not None:
        cmd.extend(["--offset", str(offset_ms)])
    
    return run_command(cmd, "Creating overlay video")


def open_video(video_path):
    """Open video in default player."""
    if sys.platform == 'win32':
        os.startfile(str(video_path))
    elif sys.platform == 'darwin':
        subprocess.run(['open', str(video_path)])
    else:
        subprocess.run(['xdg-open', str(video_path)])


def main():
    parser = argparse.ArgumentParser(
        description="Process pothole detection session - copy, convert, and create overlay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python process_session.py session_20260201_012024
    python process_session.py latest
    python process_session.py latest --offset 3000
    python process_session.py latest --no-copy
    python process_session.py latest --open
        """
    )
    
    parser.add_argument('session', type=str, 
                        help='Session name or "latest" for most recent')
    parser.add_argument('--offset', type=int, default=None,
                        help='Manual sync offset in ms (default: auto-detect)')
    parser.add_argument('--no-copy', action='store_true',
                        help='Skip copying from Pi (use local data)')
    parser.add_argument('--open', action='store_true',
                        help='Open overlay video when done')
    parser.add_argument('--pi-host', type=str, default=PI_HOST,
                        help=f'Raspberry Pi IP address (default: {PI_HOST})')
    
    args = parser.parse_args()
    
    # Use command-line host or default
    pi_host = args.pi_host
    
    print("\n" + "=" * 60)
    print("  POTHOLE SESSION PROCESSOR")
    print("=" * 60)
    
    # Ensure local data directory exists
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find ffmpeg
    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        print("\nERROR: ffmpeg not found. Please install ffmpeg.")
        print("  Windows: winget install ffmpeg")
        sys.exit(1)
    print(f"\nUsing ffmpeg: {ffmpeg_path}")
    
    # Determine session name
    session_name = args.session
    if session_name.lower() == 'latest':
        if not args.no_copy:
            print("\nFinding latest session on Pi...")
            sessions = list_remote_sessions(pi_host)
            if not sessions:
                print("ERROR: No sessions found on Pi")
                sys.exit(1)
            session_name = sessions[-1]
            print(f"  Latest session: {session_name}")
        else:
            # Find latest local session
            local_sessions = sorted([d.name for d in LOCAL_DATA_DIR.iterdir() 
                                    if d.is_dir() and d.name.startswith('session_')])
            if not local_sessions:
                print("ERROR: No local sessions found")
                sys.exit(1)
            session_name = local_sessions[-1]
            print(f"  Latest local session: {session_name}")
    
    session_dir = LOCAL_DATA_DIR / session_name
    
    # Step 1: Copy from Pi
    if not args.no_copy:
        print(f"\n[1/3] COPYING FROM PI")
        session_dir = copy_from_pi(session_name, LOCAL_DATA_DIR, pi_host)
        if not session_dir:
            print("ERROR: Failed to copy session")
            sys.exit(1)
    else:
        print(f"\n[1/3] SKIPPING COPY (using local data)")
        if not session_dir.exists():
            print(f"ERROR: Session not found: {session_dir}")
            sys.exit(1)
    
    # Step 2: Convert to MP4
    print(f"\n[2/3] CONVERTING VIDEO")
    if not convert_to_mp4(session_dir, ffmpeg_path):
        print("WARNING: Video conversion failed, continuing anyway")
    
    # Step 3: Create overlay
    print(f"\n[3/3] CREATING OVERLAY")
    if not create_overlay(session_dir, args.offset):
        print("ERROR: Overlay creation failed")
        sys.exit(1)
    
    # Summary
    overlay_path = session_dir / "video" / "capture_overlay.mp4"
    print("\n" + "=" * 60)
    print("  PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nSession: {session_name}")
    print(f"Location: {session_dir}")
    print(f"\nOutput files:")
    print(f"  Video:   {session_dir / 'video' / 'capture.mp4'}")
    print(f"  Overlay: {overlay_path}")
    print(f"  Accel:   {session_dir / 'accelerometer' / 'accel_log.csv'}")
    print(f"  Labels:  {session_dir / 'labels' / 'events.json'}")
    
    # Open video if requested
    if args.open and overlay_path.exists():
        print("\nOpening overlay video...")
        open_video(overlay_path)


if __name__ == "__main__":
    main()
