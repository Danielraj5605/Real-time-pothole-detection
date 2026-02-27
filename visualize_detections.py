"""
Visualize Pothole Detection Results
Creates comprehensive graphs showing accelerometer data, vision detections, and fusion results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for better visuals
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 16

# Load detection report
print("Loading detection report...")
with open('Datasets/live data/session_20260211_171502/pothole_detection_report.json', 'r') as f:
    report = json.load(f)

# Load accelerometer data
print("Loading accelerometer data...")
accel_df = pd.read_csv('Datasets/live data/session_20260211_171502/accelerometer/accel_log.csv')

# Convert to seconds
accel_df['time_s'] = accel_df['timestamp_ms'] / 1000

# Calculate magnitude
accel_df['magnitude'] = np.sqrt(accel_df['ay']**2 + accel_df['az']**2)

# Extract events data
events = report['events']
event_times = [e['timestamp_ms'] / 1000 for e in events]
event_accel_peak = [e['accel_peak_g'] for e in events]
event_accel_rms = [e['accel_rms_g'] for e in events]
event_vision_conf = [e['vision_confidence'] if e['has_vision_detection'] else 0 for e in events]
event_fusion_score = [e['fusion_score'] for e in events]
event_severity = [e['fusion_severity'] for e in events]

# Create color mapping for severity
severity_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
event_colors = [severity_colors[s] for s in event_severity]

# Create the visualization with improved spacing (4 rows instead of 5)
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.25, 
                      left=0.08, right=0.95, top=0.95, bottom=0.06)

# ============================================================================
# 1. Accelerometer Magnitude Over Time
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(accel_df['time_s'], accel_df['magnitude'], 
         color='steelblue', linewidth=0.5, alpha=0.7, label='Acceleration Magnitude')

# Mark detected events
for i, (t, peak, severity) in enumerate(zip(event_times, event_accel_peak, event_severity)):
    color = severity_colors[severity]
    ax1.scatter(t, peak, color=color, s=50, alpha=0.6, zorder=5)

# Threshold lines
ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='MEDIUM Threshold (0.5G)')
ax1.axhline(y=1.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='HIGH Threshold (1.5G)')

ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Acceleration Magnitude (G)', fontsize=12, fontweight='bold')
ax1.set_title('Accelerometer Data - Full Session', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# ============================================================================
# 2. Pothole Events Timeline
# ============================================================================
ax2 = fig.add_subplot(gs[1, :])

# Create event markers
for i, (t, severity) in enumerate(zip(event_times, event_severity)):
    color = severity_colors[severity]
    ax2.scatter(t, 1, color=color, s=100, alpha=0.7, marker='|', linewidths=3)

ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Events', fontsize=12, fontweight='bold')
ax2.set_title('Pothole Detection Timeline (Color = Severity)', fontsize=14, fontweight='bold')
ax2.set_ylim(0.5, 1.5)
ax2.set_yticks([])
ax2.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', label='HIGH'),
    Patch(facecolor='orange', label='MEDIUM'),
    Patch(facecolor='green', label='LOW')
]
ax2.legend(handles=legend_elements, loc='upper right')

# ============================================================================
# 3. Accelerometer Peak vs RMS
# ============================================================================
ax3 = fig.add_subplot(gs[2, 0])
scatter = ax3.scatter(event_accel_peak, event_accel_rms, 
                     c=event_colors, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax3.axvline(x=1.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_xlabel('Peak Acceleration (G)', fontsize=11, fontweight='bold')
ax3.set_ylabel('RMS Vibration (G)', fontsize=11, fontweight='bold')
ax3.set_title('Accelerometer: Peak vs RMS', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ============================================================================
# 4. Vision Confidence Distribution
# ============================================================================
ax4 = fig.add_subplot(gs[2, 1])
vision_detected = [e for e in events if e['has_vision_detection']]
vision_confidences = [e['vision_confidence'] for e in vision_detected]

ax4.hist(vision_confidences, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax4.axvline(x=np.mean(vision_confidences), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {np.mean(vision_confidences):.2f}')
ax4.set_xlabel('Vision Confidence', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title(f'Vision Confidence Distribution (n={len(vision_detected)})', 
             fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# ============================================================================
# 5. Fusion Score Over Time
# ============================================================================
ax5 = fig.add_subplot(gs[3, :])
ax5.scatter(event_times, event_fusion_score, c=event_colors, s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
ax5.plot(event_times, event_fusion_score, color='gray', linewidth=1, alpha=0.3, zorder=1)

ax5.axhline(y=0.4, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='MEDIUM Threshold')
ax5.axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='HIGH Threshold')

ax5.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Fusion Score', fontsize=12, fontweight='bold')
ax5.set_title('Multimodal Fusion Score Over Time', fontsize=14, fontweight='bold')
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3)

# ============================================================================
# Main Title
# ============================================================================
fig.suptitle('Real-Time Pothole Detection System - Multimodal Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

# Save the figure with proper spacing
output_path = 'Datasets/live data/session_20260211_171502/detection_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
print(f"\n✅ Visualization saved to: {output_path}")

# Show the plot
plt.show()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
