#!/usr/bin/env python3
"""
Training Progress Monitor

Monitors the training progress by reading the latest training directory.

Usage:
    python scripts/monitor_training.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_latest_training():
    """Find the most recent training directory"""
    training_dir = project_root / "models" / "yolo_training"
    if not training_dir.exists():
        return None
    
    # Find all training directories
    dirs = [d for d in training_dir.iterdir() if d.is_dir() and d.name.startswith("pothole_")]
    if not dirs:
        return None
    
    # Return the most recent one
    return max(dirs, key=lambda d: d.stat().st_mtime)


def read_results(training_path):
    """Read training results from results.csv"""
    results_file = training_path / "results.csv"
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None
            
            # Get header and last line
            header = lines[0].strip().split(',')
            last_line = lines[-1].strip().split(',')
            
            # Create dict
            data = {}
            for i, key in enumerate(header):
                if i < len(last_line):
                    try:
                        data[key.strip()] = float(last_line[i].strip())
                    except:
                        data[key.strip()] = last_line[i].strip()
            
            return data
    except Exception as e:
        print(f"Error reading results: {e}")
        return None


def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def monitor_training(interval=30):
    """Monitor training progress"""
    print("="*60)
    print("TRAINING PROGRESS MONITOR")
    print("="*60)
    print(f"Checking every {interval} seconds. Press Ctrl+C to stop.\n")
    
    training_path = find_latest_training()
    if not training_path:
        print("[ERROR] No training directory found!")
        print("Make sure training has started.")
        return
    
    print(f"Monitoring: {training_path.name}\n")
    
    start_time = time.time()
    last_epoch = 0
    
    try:
        while True:
            results = read_results(training_path)
            
            if results:
                epoch = int(results.get('epoch', 0))
                
                # Calculate progress
                total_epochs = 100  # Default, adjust if needed
                progress = (epoch / total_epochs) * 100
                
                # Estimate time remaining
                elapsed = time.time() - start_time
                if epoch > 0:
                    time_per_epoch = elapsed / epoch
                    remaining_epochs = total_epochs - epoch
                    eta = time_per_epoch * remaining_epochs
                else:
                    eta = 0
                
                # Clear screen (optional)
                print("\033[H\033[J", end="")  # Clear screen
                
                print("="*60)
                print(f"TRAINING PROGRESS - {datetime.now().strftime('%H:%M:%S')}")
                print("="*60)
                print(f"\nTraining: {training_path.name}")
                print(f"\nEpoch: {epoch}/{total_epochs} ({progress:.1f}%)")
                print(f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}")
                
                # Progress bar
                bar_length = 40
                filled = int(bar_length * progress / 100)
                bar = '#' * filled + '-' * (bar_length - filled)
                print(f"\n[{bar}] {progress:.1f}%")
                
                # Metrics
                print(f"\n--- Current Metrics ---")
                print(f"Box Loss:   {results.get('train/box_loss', 0):.4f}")
                print(f"Class Loss: {results.get('train/cls_loss', 0):.4f}")
                print(f"DFL Loss:   {results.get('train/dfl_loss', 0):.4f}")
                
                if 'metrics/mAP50(B)' in results:
                    print(f"\nmAP@0.5:    {results.get('metrics/mAP50(B)', 0):.4f}")
                if 'metrics/mAP50-95(B)' in results:
                    print(f"mAP@0.5:0.95: {results.get('metrics/mAP50-95(B)', 0):.4f}")
                
                # Check if training completed
                if epoch >= total_epochs:
                    print("\n" + "="*60)
                    print("[SUCCESS] Training completed!")
                    print("="*60)
                    
                    weights_path = training_path / "weights" / "best.pt"
                    if weights_path.exists():
                        print(f"\nBest weights: {weights_path}")
                    break
                
                # New epoch notification
                if epoch > last_epoch:
                    print(f"\n[INFO] Completed epoch {epoch}")
                    last_epoch = epoch
                
            else:
                print(f"Waiting for results... ({datetime.now().strftime('%H:%M:%S')})")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] Monitoring stopped by user")
        print(f"Training directory: {training_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor YOLOv8 training progress")
    parser.add_argument('--interval', type=int, default=30, 
                       help='Check interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    monitor_training(interval=args.interval)
