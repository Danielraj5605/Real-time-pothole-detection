"""
YOLOv5 Pothole Detection Model Testing Script
==============================================
Test the trained model on test dataset and visualize results
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from datetime import datetime

class PotholeDetector:
    """Pothole detection inference class"""
    
    def __init__(self, weights_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize detector
        
        Args:
            weights_path: Path to trained weights
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        print(f"Loading model from {self.weights_path}...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                     path=str(self.weights_path), force_reload=False)
        
        # Set thresholds
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        
        # Class names
        self.classes = ['low', 'medium', 'high']
        self.colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 165, 255),  # Orange
            'high': (0, 0, 255)       # Red
        }
        
        print("Model loaded successfully!")
    
    def detect(self, image_path):
        """
        Detect potholes in an image
        
        Args:
            image_path: Path to image file
            
        Returns:
            results: Detection results
        """
        results = self.model(str(image_path))
        return results
    
    def visualize_detection(self, image_path, save_path=None, show=True):
        """
        Visualize detection results
        
        Args:
            image_path: Path to image
            save_path: Path to save annotated image
            show: Whether to display image
        """
        # Read image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect
        results = self.detect(image_path)
        
        # Parse results
        detections = results.pandas().xyxy[0]
        
        # Draw detections
        for idx, det in detections.iterrows():
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            conf = det['confidence']
            cls = det['name']
            
            # Get color based on severity
            color = self.colors.get(cls, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls.upper()}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for text
            cv2.rectangle(img_rgb, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(img_rgb, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title(f"Detections: {len(detections)}")
            plt.tight_layout()
            plt.show()
        
        # Save
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            print(f"Saved annotated image to {save_path}")
        
        return detections


def evaluate_on_test_set(weights_path, test_images_dir, output_dir):
    """
    Evaluate model on test set
    
    Args:
        weights_path: Path to trained weights
        test_images_dir: Directory containing test images
        output_dir: Directory to save results
    """
    print("=" * 70)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize detector
    detector = PotholeDetector(weights_path)
    
    # Get test images
    test_images = list(Path(test_images_dir).glob("*.jpg"))
    print(f"\nFound {len(test_images)} test images")
    
    # Process each image
    all_results = []
    
    for idx, img_path in enumerate(test_images[:20]):  # Process first 20 for demo
        print(f"\nProcessing {idx+1}/{min(20, len(test_images))}: {img_path.name}")
        
        # Detect
        detections = detector.visualize_detection(
            img_path, 
            save_path=output_dir / f"result_{img_path.name}",
            show=False
        )
        
        # Store results
        result = {
            'image': img_path.name,
            'num_detections': len(detections),
            'detections': detections.to_dict('records') if len(detections) > 0 else []
        }
        all_results.append(result)
        
        print(f"  Detections: {len(detections)}")
        if len(detections) > 0:
            for _, det in detections.iterrows():
                print(f"    - {det['name'].upper()}: {det['confidence']:.3f}")
    
    # Save results to JSON
    results_file = output_dir / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Summary statistics
    total_detections = sum(r['num_detections'] for r in all_results)
    avg_detections = total_detections / len(all_results) if all_results else 0
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total images processed: {len(all_results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    print("=" * 70)


def test_single_image(weights_path, image_path):
    """
    Test model on a single image
    
    Args:
        weights_path: Path to trained weights
        image_path: Path to test image
    """
    print("=" * 70)
    print("TESTING ON SINGLE IMAGE")
    print("=" * 70)
    
    detector = PotholeDetector(weights_path)
    
    print(f"\nImage: {image_path}")
    detections = detector.visualize_detection(image_path, show=True)
    
    print(f"\nDetections: {len(detections)}")
    for idx, det in detections.iterrows():
        print(f"  {idx+1}. {det['name'].upper()} - Confidence: {det['confidence']:.3f}")
        print(f"     BBox: ({det['xmin']:.0f}, {det['ymin']:.0f}, {det['xmax']:.0f}, {det['ymax']:.0f})")
    
    print("=" * 70)


def benchmark_speed(weights_path, test_image):
    """
    Benchmark inference speed
    
    Args:
        weights_path: Path to trained weights
        test_image: Path to test image
    """
    import time
    
    print("=" * 70)
    print("BENCHMARKING INFERENCE SPEED")
    print("=" * 70)
    
    detector = PotholeDetector(weights_path)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        _ = detector.detect(test_image)
    
    # Benchmark
    print("Running benchmark (100 iterations)...")
    times = []
    
    for i in range(100):
        start = time.time()
        _ = detector.detect(test_image)
        end = time.time()
        times.append(end - start)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/100")
    
    # Statistics
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    fps = 1000 / avg_time
    
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Average inference time: {avg_time:.2f} ms (± {std_time:.2f} ms)")
    print(f"Min inference time: {min_time:.2f} ms")
    print(f"Max inference time: {max_time:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print("=" * 70)
    
    # Note for Raspberry Pi
    print("\nNOTE: These results are for your current hardware.")
    print("On Raspberry Pi 4, expect ~2-5 FPS with YOLOv5s model.")
    print("For better performance on Pi, consider:")
    print("  - Using YOLOv5n (nano) model")
    print("  - Reducing input image size")
    print("  - Using TensorRT or ONNX Runtime")
    print("=" * 70)


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Pothole Detection Model')
    parser.add_argument('--weights', type=str, default='weights/pothole_best.pt',
                       help='Path to trained weights')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'test_set', 'benchmark'],
                       help='Testing mode')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single test image')
    parser.add_argument('--test_dir', type=str, default='Datasets/test/images',
                       help='Path to test images directory')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if weights exist
    if not Path(args.weights).exists():
        print(f"ERROR: Weights file not found: {args.weights}")
        print("Please train the model first using train_yolov5.py")
        return
    
    if args.mode == 'single':
        if args.image is None:
            # Use first image from test set
            test_images = list(Path(args.test_dir).glob("*.jpg"))
            if test_images:
                args.image = str(test_images[0])
            else:
                print("ERROR: No test images found and no image specified")
                return
        
        test_single_image(args.weights, args.image)
    
    elif args.mode == 'test_set':
        evaluate_on_test_set(args.weights, args.test_dir, args.output)
    
    elif args.mode == 'benchmark':
        if args.image is None:
            test_images = list(Path(args.test_dir).glob("*.jpg"))
            if test_images:
                args.image = str(test_images[0])
            else:
                print("ERROR: No test images found")
                return
        
        benchmark_speed(args.weights, args.image)


if __name__ == "__main__":
    main()
