"""
Refinement Test Script
Attempts to tighten loose YOLO bounding boxes using computer vision techniques.
"""
import cv2
import numpy as np

def refine_pothole_bbox(image, bbox):
    """
    Refines a bounding box to focus on the most likely pothole region within it.
    """
    x1, y1, x2, y2 = bbox
    
    # 1. Extract the region of interest (ROI) from the bounding box
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return bbox

    # 2. Convert to gray and apply Gaussian blur to remove noise
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Potholes are usually darker than the road. 
    # Invert so potholes become bright blobs.
    inverted = cv2.bitwise_not(blurred)
    
    # 4. Adaptive Thresholding to isolate "dark patches" locally
    # This helps even if the road has shadows
    thresh = cv2.adaptiveThreshold(
        inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, -5
    )
    
    # 5. Find contours of these dark patches
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return bbox

    # 6. Filter contours
    possible_potholes = []
    roi_area = (x2 - x1) * (y2 - y1)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter 1: Too small (noise) or too big (entire road shadow)
        if area < roi_area * 0.01 or area > roi_area * 0.8:
            continue
            
        # Filter 2: Aspect ratio (potholes generally aren't thin lines)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect = float(bw) / bh
        if aspect > 4 or aspect < 0.25:
            continue
            
        possible_potholes.append(cnt)

    if not possible_potholes:
        # Fallback: If no distinct hole found, shrink to the center 50% 
        # (heuristic: detection is usually centered on the object)
        center_x = (x2 - x1) // 2
        center_y = (y2 - y1) // 2
        w_new = (x2 - x1) // 2
        h_new = (y2 - y1) // 2
        return (x1 + center_x - w_new//2, y1 + center_y - h_new//2, 
                x1 + center_x + w_new//2, y1 + center_y + h_new//2)

    # 7. Merge relevant contours to form a new tight box
    # Combine all valid "dark spots" into one bounding box
    combined_cnt = np.concatenate(possible_potholes)
    rx, ry, rw, rh = cv2.boundingRect(combined_cnt)
    
    # Return new coordinates relative to the original image
    return (x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh)

# --- Test Execution ---
# Load a sample frame
img = cv2.imread('Datasets/live data/session_20260211_171502/sample_frame_60.jpg')
if img is None:
    print("Error: Could not load sample frame.")
    exit()

# Simulate a "Bad" YOLO detection (The large blue box we saw)
# Based on your video report, detections were often ~80% of frame
h, w = img.shape[:2]
bad_bbox = (int(w*0.1), int(h*0.2), int(w*0.9), int(h*0.9)) # A huge box

# Apply refinement
refined_bbox = refine_pothole_bbox(img, bad_bbox)

# Visualize
debug_img = img.copy()
# Draw "Original" Bad Box in Red (Thin)
cv2.rectangle(debug_img, (bad_bbox[0], bad_bbox[1]), (bad_bbox[2], bad_bbox[3]), (0, 0, 255), 2)
# Draw "Refined" Box in Green (Thick)
cv2.rectangle(debug_img, (refined_bbox[0], refined_bbox[1]), (refined_bbox[2], refined_bbox[3]), (0, 255, 0), 4)

cv2.imwrite('Datasets/live data/session_20260211_171502/refinement_test.jpg', debug_img)
print("Test complete. Saved 'refinement_test.jpg'")
print(f"Original Area: {(bad_bbox[2]-bad_bbox[0])*(bad_bbox[3]-bad_bbox[1])}")
print(f"Refined Area:  {(refined_bbox[2]-refined_bbox[0])*(refined_bbox[3]-refined_bbox[1])}")
