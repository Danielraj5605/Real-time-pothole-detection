"""
Label Preparation and Validation Script
========================================
Helper script to prepare and validate YOLO format labels for pothole detection
"""

import cv2
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


class LabelValidator:
    """Validate and visualize YOLO format labels"""
    
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.classes = ['low', 'medium', 'high']
        self.colors = {
            0: (0, 255, 0),      # Green - low
            1: (0, 165, 255),    # Orange - medium
            2: (0, 0, 255)       # Red - high
        }
    
    def parse_yolo_label(self, label_path, img_width, img_height):
        """
        Parse YOLO format label file
        
        Args:
            label_path: Path to label file
            img_width: Image width
            img_height: Image height
            
        Returns:
            List of bounding boxes [(class_id, x1, y1, x2, y2), ...]
        """
        boxes = []
        
        if not label_path.exists():
            return boxes
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to pixel coordinates
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                boxes.append((class_id, x1, y1, x2, y2))
        
        return boxes
    
    def visualize_labels(self, image_name, save_path=None):
        """
        Visualize labels on image
        
        Args:
            image_name: Name of image file
            save_path: Path to save visualization
        """
        # Load image
        img_path = self.images_dir / image_name
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"ERROR: Could not load image {img_path}")
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Load labels
        label_path = self.labels_dir / f"{Path(image_name).stem}.txt"
        boxes = self.parse_yolo_label(label_path, w, h)
        
        # Draw boxes
        for class_id, x1, y1, x2, y2 in boxes:
            color = self.colors.get(class_id, (255, 255, 255))
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = self.classes[class_id].upper()
            cv2.putText(img_rgb, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"{image_name} - {len(boxes)} annotations")
        
        # Legend
        patches = [mpatches.Patch(color=np.array(self.colors[i])/255, 
                                 label=self.classes[i].upper()) 
                  for i in range(len(self.classes))]
        plt.legend(handles=patches, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def validate_dataset(self):
        """
        Validate entire dataset
        
        Returns:
            Dictionary with validation statistics
        """
        print("=" * 70)
        print("VALIDATING DATASET")
        print("=" * 70)
        
        # Get all images
        image_files = list(self.images_dir.glob("*.jpg"))
        print(f"\nTotal images: {len(image_files)}")
        
        # Statistics
        stats = {
            'total_images': len(image_files),
            'images_with_labels': 0,
            'images_without_labels': 0,
            'total_annotations': 0,
            'class_distribution': {cls: 0 for cls in self.classes},
            'invalid_labels': [],
            'empty_labels': []
        }
        
        # Validate each image
        for img_path in image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                stats['images_without_labels'] += 1
                stats['empty_labels'].append(img_path.name)
                continue
            
            # Load image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            # Parse labels
            boxes = self.parse_yolo_label(label_path, w, h)
            
            if len(boxes) == 0:
                stats['images_without_labels'] += 1
                stats['empty_labels'].append(img_path.name)
            else:
                stats['images_with_labels'] += 1
                stats['total_annotations'] += len(boxes)
                
                # Count classes
                for class_id, _, _, _, _ in boxes:
                    if 0 <= class_id < len(self.classes):
                        stats['class_distribution'][self.classes[class_id]] += 1
        
        # Print statistics
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        print(f"Images with labels: {stats['images_with_labels']}")
        print(f"Images without labels: {stats['images_without_labels']}")
        print(f"Total annotations: {stats['total_annotations']}")
        
        print("\nClass Distribution:")
        for cls, count in stats['class_distribution'].items():
            percentage = (count / stats['total_annotations'] * 100) if stats['total_annotations'] > 0 else 0
            print(f"  {cls.upper()}: {count} ({percentage:.1f}%)")
        
        if stats['empty_labels']:
            print(f"\nWARNING: {len(stats['empty_labels'])} images without labels")
            print("First 5 examples:")
            for name in stats['empty_labels'][:5]:
                print(f"  - {name}")
        
        print("=" * 70)
        
        return stats


def create_sample_labels():
    """
    Create sample label files for demonstration
    This is a helper function to show the expected format
    """
    print("=" * 70)
    print("SAMPLE LABEL FORMAT")
    print("=" * 70)
    
    sample_label = """# YOLO Format Label File
# Format: <class_id> <x_center> <y_center> <width> <height>
# All values are normalized to [0, 1]
# 
# Class IDs:
#   0 = low severity
#   1 = medium severity
#   2 = high severity
#
# Example annotations:

0 0.5 0.5 0.3 0.2
1 0.7 0.3 0.15 0.15
2 0.2 0.8 0.25 0.18
"""
    
    print(sample_label)
    print("\nTo create labels:")
    print("1. Use annotation tools like LabelImg, Roboflow, or CVAT")
    print("2. Export in YOLO format")
    print("3. Place label files in 'Datasets/labels/' directory")
    print("4. Label filename should match image filename (e.g., image.jpg -> image.txt)")
    print("=" * 70)


def check_label_format(label_file):
    """
    Check if a label file is in correct YOLO format
    
    Args:
        label_file: Path to label file
    """
    print(f"\nChecking: {label_file}")
    
    issues = []
    
    with open(label_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            # Check number of values
            if len(parts) != 5:
                issues.append(f"Line {line_num}: Expected 5 values, got {len(parts)}")
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Validate ranges
                if class_id < 0 or class_id > 2:
                    issues.append(f"Line {line_num}: Invalid class_id {class_id} (should be 0-2)")
                
                if not (0 <= x_center <= 1):
                    issues.append(f"Line {line_num}: x_center {x_center} out of range [0,1]")
                
                if not (0 <= y_center <= 1):
                    issues.append(f"Line {line_num}: y_center {y_center} out of range [0,1]")
                
                if not (0 < width <= 1):
                    issues.append(f"Line {line_num}: width {width} out of range (0,1]")
                
                if not (0 < height <= 1):
                    issues.append(f"Line {line_num}: height {height} out of range (0,1]")
                
            except ValueError as e:
                issues.append(f"Line {line_num}: Invalid number format - {e}")
    
    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    else:
        print("  ✓ Format is correct")
        return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Label Validation and Preparation')
    parser.add_argument('--mode', type=str, default='validate',
                       choices=['validate', 'visualize', 'sample', 'check'],
                       help='Operation mode')
    parser.add_argument('--images', type=str, default='Datasets/images',
                       help='Path to images directory')
    parser.add_argument('--labels', type=str, default='Datasets/labels',
                       help='Path to labels directory')
    parser.add_argument('--image', type=str, default=None,
                       help='Specific image to visualize')
    parser.add_argument('--label_file', type=str, default=None,
                       help='Specific label file to check')
    
    args = parser.parse_args()
    
    if args.mode == 'validate':
        validator = LabelValidator(args.images, args.labels)
        validator.validate_dataset()
    
    elif args.mode == 'visualize':
        validator = LabelValidator(args.images, args.labels)
        
        if args.image:
            validator.visualize_labels(args.image)
        else:
            # Visualize first 5 images
            images = list(Path(args.images).glob("*.jpg"))[:5]
            for img in images:
                validator.visualize_labels(img.name)
    
    elif args.mode == 'sample':
        create_sample_labels()
    
    elif args.mode == 'check':
        if args.label_file:
            check_label_format(args.label_file)
        else:
            print("Please specify --label_file to check")


if __name__ == "__main__":
    main()
