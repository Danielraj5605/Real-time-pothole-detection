"""
Quick visualization of processing results
"""
import json
import matplotlib.pyplot as plt
from collections import Counter

# Load report
with open('Datasets/live data/session_20260211_171502/pothole_detection_report.json', 'r') as f:
    report = json.load(f)

events = report['events']

# Extract data
timestamps = [e['timestamp_ms'] / 1000 for e in events]  # Convert to seconds
accel_peaks = [e['accel_peak_g'] for e in events]
fusion_scores = [e['fusion_score'] for e in events]
severities = [e['fusion_severity'] for e in events]
has_vision = [e['has_vision_detection'] for e in events]

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Live Session Processing Results - session_20260211_171502', fontsize=16, fontweight='bold')

# 1. Accelerometer peaks over time
ax1 = axes[0, 0]
colors = ['red' if v else 'blue' for v in has_vision]
ax1.scatter(timestamps, accel_peaks, c=colors, alpha=0.6, s=30)
ax1.axhline(y=0.5, color='orange', linestyle='--', label='Threshold (0.5G)')
ax1.axhline(y=1.5, color='red', linestyle='--', label='High Severity (1.5G)')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Accelerometer Peak (G)')
ax1.set_title('Accelerometer Peaks Over Time')
ax1.legend(['Vision Detected', 'No Vision', 'Threshold', 'High Severity'])
ax1.grid(True, alpha=0.3)

# 2. Fusion scores over time
ax2 = axes[0, 1]
severity_colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'yellow'}
colors = [severity_colors[s] for s in severities]
ax2.scatter(timestamps, fusion_scores, c=colors, alpha=0.6, s=30)
ax2.axhline(y=0.7, color='red', linestyle='--', label='HIGH threshold')
ax2.axhline(y=0.4, color='orange', linestyle='--', label='MEDIUM threshold')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Fusion Score')
ax2.set_title('Fusion Scores Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Severity distribution
ax3 = axes[1, 0]
severity_counts = Counter(severities)
colors_bar = [severity_colors[s] for s in ['HIGH', 'MEDIUM', 'LOW']]
ax3.bar(severity_counts.keys(), severity_counts.values(), color=colors_bar, alpha=0.7)
ax3.set_ylabel('Count')
ax3.set_title('Severity Distribution')
ax3.grid(True, axis='y', alpha=0.3)
for i, (k, v) in enumerate(severity_counts.items()):
    ax3.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 4. Vision detection stats
ax4 = axes[1, 1]
vision_counts = {'Vision Detected': sum(has_vision), 'No Vision': len(has_vision) - sum(has_vision)}
ax4.pie(vision_counts.values(), labels=vision_counts.keys(), autopct='%1.1f%%', 
        colors=['green', 'gray'], startangle=90)
ax4.set_title('Vision Detection Rate')

plt.tight_layout()
plt.savefig('Datasets/live data/session_20260211_171502/processing_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved: processing_visualization.png")
print(f"\nSummary:")
print(f"  Total events: {len(events)}")
print(f"  Vision detected: {sum(has_vision)} ({sum(has_vision)/len(events)*100:.1f}%)")
print(f"  Severity: HIGH={severity_counts['HIGH']}, MEDIUM={severity_counts['MEDIUM']}, LOW={severity_counts['LOW']}")
print(f"  Avg accel peak: {sum(accel_peaks)/len(accel_peaks):.2f}G")
print(f"  Avg fusion score: {sum(fusion_scores)/len(fusion_scores):.2f}")
