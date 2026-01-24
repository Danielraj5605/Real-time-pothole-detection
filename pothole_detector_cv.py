"""
Pothole Detection using Digital Image Processing
=================================================
Classical Computer Vision approach without deep learning
Uses edge detection, contour analysis, and morphological operations
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import json


class PotholeDetectorCV:
    """Pothole detection using classical computer vision techniques"""
    
    def __init__(self):
        """Initialize detector with default parameters"""
        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        
        # Morphological operation kernel size
        self.kernel_size = 5
        
        # Contour filtering parameters
        self.min_area = 500  # Minimum pothole area in pixels
        self.max_area = 50000  # Maximum pothole area
        
        # Severity thresholds (based on area)
        self.low_threshold = 2000
        self.medium_threshold = 5000
        
        # Colors for visualization
        self.colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 165, 255),  # Orange
            'high': (0, 0, 255)       # Red
        }
    
    def preprocess_image(self, image):
        """
        Preprocess image for pothole detection
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_edges(self, preprocessed):
        """
        Detect edges using Canny edge detector
        
        Args:
            preprocessed: Preprocessed grayscale image
            
        Returns:
            Edge map
        """
        edges = cv2.Canny(preprocessed, self.canny_low, self.canny_high)
        return edges
    
    def apply_morphology(self, edges):
        """
        Apply morphological operations to connect edges
        
        Args:
            edges: Edge map
            
        Returns:
            Processed edge map
        """
        # Create kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (self.kernel_size, self.kernel_size))
        
        # Closing operation to connect nearby edges
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Dilation to make edges more prominent
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        return dilated
    
    def find_potholes(self, processed_edges):
        """
        Find pothole contours
        
        Args:
            processed_edges: Processed edge map
            
        Returns:
            List of pothole contours
        """
        # Find contours
        contours, _ = cv2.findContours(processed_edges, 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        potholes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # Additional shape analysis
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    
                    # Potholes are typically somewhat circular (0.3 - 1.0)
                    if 0.3 < circularity < 1.0:
                        potholes.append(contour)
        
        return potholes
    
    def classify_severity(self, contour):
        """
        Classify pothole severity based on area
        
        Args:
            contour: Pothole contour
            
        Returns:
            Severity level ('low', 'medium', 'high')
        """
        area = cv2.contourArea(contour)
        
        if area < self.low_threshold:
            return 'low'
        elif area < self.medium_threshold:
            return 'medium'
        else:
            return 'high'
    
    def detect(self, image_path):
        """
        Detect potholes in an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with detection results
        """
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Detection pipeline
        preprocessed = self.preprocess_image(image)
        edges = self.detect_edges(preprocessed)
        processed = self.apply_morphology(edges)
        potholes = self.find_potholes(processed)
        
        # Analyze each pothole
        results = []
        for i, contour in enumerate(potholes):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Classify severity
            severity = self.classify_severity(contour)
            
            results.append({
                'id': i + 1,
                'bbox': (x, y, w, h),
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'severity': severity,
                'contour': contour
            })
        
        return {
            'image_path': str(image_path),
            'num_potholes': len(results),
            'potholes': results,
            'processed_images': {
                'original': image,
                'preprocessed': preprocessed,
                'edges': edges,
                'processed': processed
            }
        }
    
    def visualize(self, detection_results, save_path=None, show=True):
        """
        Visualize detection results
        
        Args:
            detection_results: Results from detect()
            save_path: Path to save visualization
            show: Whether to display the image
        """
        image = detection_results['processed_images']['original'].copy()
        
        # Draw each pothole
        for pothole in detection_results['potholes']:
            x, y, w, h = pothole['bbox']
            severity = pothole['severity']
            color = self.colors[severity]
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw contour
            cv2.drawContours(image, [pothole['contour']], -1, color, 2)
            
            # Add label
            label = f"{severity.upper()} #{pothole['id']}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for text
            cv2.rectangle(image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Text
            cv2.putText(image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add area info
            area_text = f"Area: {int(pothole['area'])} px"
            cv2.putText(image, area_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add summary
        summary = f"Potholes Detected: {detection_results['num_potholes']}"
        cv2.putText(image, summary, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display
        if show:
            plt.figure(figsize=(15, 10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"Pothole Detection Results - {detection_results['num_potholes']} detected")
            plt.tight_layout()
            plt.show()
        
        # Save
        if save_path:
            cv2.imwrite(str(save_path), image)
            print(f"Saved visualization to: {save_path}")
        
        return image
    
    def visualize_pipeline(self, detection_results, save_path=None):
        """
        Visualize the entire processing pipeline
        
        Args:
            detection_results: Results from detect()
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        original = detection_results['processed_images']['original']
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Preprocessed
        preprocessed = detection_results['processed_images']['preprocessed']
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('Preprocessed (CLAHE)')
        axes[0, 1].axis('off')
        
        # Edges
        edges = detection_results['processed_images']['edges']
        axes[0, 2].imshow(edges, cmap='gray')
        axes[0, 2].set_title('Edge Detection (Canny)')
        axes[0, 2].axis('off')
        
        # Morphological processing
        processed = detection_results['processed_images']['processed']
        axes[1, 0].imshow(processed, cmap='gray')
        axes[1, 0].set_title('Morphological Processing')
        axes[1, 0].axis('off')
        
        # Contours overlay
        contour_img = original.copy()
        for pothole in detection_results['potholes']:
            color = self.colors[pothole['severity']]
            cv2.drawContours(contour_img, [pothole['contour']], -1, color, 2)
        axes[1, 1].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Detected Contours')
        axes[1, 1].axis('off')
        
        # Final result
        final = self.visualize(detection_results, show=False)
        axes[1, 2].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Final Result ({detection_results["num_potholes"]} potholes)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved pipeline visualization to: {save_path}")
        
        plt.show()


def process_single_image(image_path, output_dir=None):
    """
    Process a single image for pothole detection
    
    Args:
        image_path: Path to image
        output_dir: Directory to save results
    """
    print("=" * 70)
    print("POTHOLE DETECTION - Digital Image Processing")
    print("=" * 70)
    
    # Create detector
    detector = PotholeDetectorCV()
    
    # Detect potholes
    print(f"\nProcessing: {image_path}")
    results = detector.detect(image_path)
    
    # Print results
    print(f"\nDetection Results:")
    print(f"  Total potholes found: {results['num_potholes']}")
    
    if results['num_potholes'] > 0:
        print("\nDetailed Analysis:")
        for pothole in results['potholes']:
            print(f"\n  Pothole #{pothole['id']}:")
            print(f"    Severity: {pothole['severity'].upper()}")
            print(f"    Area: {pothole['area']:.0f} pixels")
            print(f"    Perimeter: {pothole['perimeter']:.0f} pixels")
            print(f"    Circularity: {pothole['circularity']:.2f}")
            print(f"    Bounding Box: {pothole['bbox']}")
    
    # Visualize
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save annotated image
        img_name = Path(image_path).stem
        result_path = output_dir / f"{img_name}_result.jpg"
        detector.visualize(results, save_path=result_path, show=True)
        
        # Save pipeline visualization
        pipeline_path = output_dir / f"{img_name}_pipeline.jpg"
        detector.visualize_pipeline(results, save_path=pipeline_path)
        
        # Save JSON results
        json_path = output_dir / f"{img_name}_results.json"
        json_results = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'num_potholes': results['num_potholes'],
            'potholes': [
                {
                    'id': p['id'],
                    'severity': p['severity'],
                    'area': float(p['area']),
                    'perimeter': float(p['perimeter']),
                    'circularity': float(p['circularity']),
                    'bbox': p['bbox']
                }
                for p in results['potholes']
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
    else:
        detector.visualize(results, show=True)
    
    print("=" * 70)
    
    return results


def process_batch(images_dir, output_dir):
    """
    Process multiple images
    
    Args:
        images_dir: Directory containing images
        output_dir: Directory to save results
    """
    print("=" * 70)
    print("BATCH POTHOLE DETECTION")
    print("=" * 70)
    
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get all images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"\nFound {len(image_files)} images")
    
    # Create detector
    detector = PotholeDetectorCV()
    
    # Process each image
    all_results = []
    
    for i, img_path in enumerate(image_files[:10], 1):  # Process first 10
        print(f"\nProcessing {i}/{min(10, len(image_files))}: {img_path.name}")
        
        try:
            results = detector.detect(img_path)
            
            # Save result
            result_path = output_dir / f"{img_path.stem}_result.jpg"
            detector.visualize(results, save_path=result_path, show=False)
            
            all_results.append({
                'image': img_path.name,
                'num_potholes': results['num_potholes'],
                'potholes': [
                    {
                        'severity': p['severity'],
                        'area': float(p['area'])
                    }
                    for p in results['potholes']
                ]
            })
            
            print(f"  Detected: {results['num_potholes']} potholes")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save summary
    summary_path = output_dir / 'batch_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    total_potholes = sum(r['num_potholes'] for r in all_results)
    avg_potholes = total_potholes / len(all_results) if all_results else 0
    
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Images processed: {len(all_results)}")
    print(f"Total potholes detected: {total_potholes}")
    print(f"Average potholes per image: {avg_potholes:.2f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pothole Detection using Digital Image Processing')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, default='cv_results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.image:
        process_single_image(args.image, args.output)
    elif args.batch:
        process_batch(args.batch, args.output)
    else:
        # Default: process first image from dataset
        images_dir = Path("Datasets/images")
        if images_dir.exists():
            images = list(images_dir.glob("*.jpg"))
            if images:
                print("No image specified, processing first image from dataset...")
                process_single_image(images[0], args.output)
            else:
                print("No images found in Datasets/images/")
        else:
            print("Usage:")
            print("  python pothole_detector_cv.py --image path/to/image.jpg")
            print("  python pothole_detector_cv.py --batch path/to/images/")


if __name__ == "__main__":
    main()
