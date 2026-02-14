"""
Quick verification of annotated video
Extracts first detection frame to verify blue boxes are visible
"""
import cv2
import sys

video_path = "Datasets/live data/session_20260211_171502/annotated_potholes.mp4"

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    sys.exit(1)

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("=" * 70)
print("  VIDEO VERIFICATION")
print("=" * 70)
print(f"Video path: {video_path}")
print(f"Total frames: {total_frames}")
print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")
print(f"Duration: {total_frames/fps:.1f} seconds")
print()

# Extract frames at different points to verify annotations
check_frames = [
    30,   # First event start (before detection)
    60,   # First event detection (should have blue box)
    90,   # First event end
    150,  # Second event detection
    300,  # Third event detection
]

print("Checking sample frames for blue boxes...")
print()

for frame_num in check_frames:
    if frame_num >= total_frames:
        continue
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Frame {frame_num}: ❌ Failed to read")
        continue
    
    # Check if frame has blue pixels (blue boxes)
    # Blue in BGR is (255, 0, 0) or close to it
    blue_mask = (frame[:, :, 0] > 200) & (frame[:, :, 1] < 50) & (frame[:, :, 2] < 50)
    blue_pixel_count = blue_mask.sum()
    
    # Check for black overlay bar at top (info overlay)
    top_bar = frame[0:60, :, :]
    dark_pixels = (top_bar.sum(axis=2) < 100).sum()
    has_overlay = dark_pixels > 10000  # Significant dark area
    
    status = "✅ HAS BLUE BOX" if blue_pixel_count > 1000 else "⚠️ No blue box"
    overlay_status = "| Overlay: ✅" if has_overlay else ""
    
    print(f"Frame {frame_num:4d}: {status} ({blue_pixel_count:6d} blue pixels) {overlay_status}")
    
    # Save sample frame
    if blue_pixel_count > 1000:
        output_path = f"Datasets/live data/session_20260211_171502/sample_frame_{frame_num}.jpg"
        cv2.imwrite(output_path, frame)
        print(f"           Saved sample: sample_frame_{frame_num}.jpg")

cap.release()

print()
print("=" * 70)
print("  VERIFICATION COMPLETE")
print("=" * 70)
print()
print("Summary:")
print(f"  ✅ Video file exists and is readable")
print(f"  ✅ {total_frames} frames at {fps} FPS")
print(f"  ✅ Duration: {total_frames/fps:.1f} seconds (~{total_frames/fps/60:.1f} minutes)")
print()
print("If blue boxes were detected in sample frames, the annotation is working!")
print("Check the saved sample frames to verify visually.")
