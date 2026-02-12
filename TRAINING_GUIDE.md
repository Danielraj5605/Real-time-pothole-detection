# YOLOv8 Pothole Detection - Training Guide

## Current Training Status

**Training Started**: February 9, 2026 at 15:14:42

**Configuration**:
- Model: YOLOv8n (nano)
- Epochs: 100
- Batch Size: 16
- Image Size: 640x640
- Dataset: 590 training images, 221 validation images

**Training Directory**: `models/yolo_training/pothole_yolov8n_20260209_151442/`

---

## Monitoring Training Progress

### Option 1: Use the Monitor Script (Recommended)
```bash
python scripts/monitor_training.py
```

This will show:
- Current epoch and progress percentage
- Loss metrics (box, class, DFL)
- mAP scores (when available)
- Estimated time remaining
- Progress bar

### Option 2: Check Results Manually
View the results file:
```bash
type models\yolo_training\pothole_yolov8n_20260209_151442\results.csv
```

### Option 3: View Training Plots
After a few epochs, check the generated plots:
- `models/yolo_training/pothole_yolov8n_20260209_151442/results.png`
- `models/yolo_training/pothole_yolov8n_20260209_151442/confusion_matrix.png`

---

## Expected Timeline

**Estimated Duration**: 75-85 minutes (on CPU)
- Each epoch: ~45-50 seconds
- 100 epochs total
- Early stopping may reduce this if model converges early

**Checkpoints**:
- Saved every 10 epochs to `weights/epoch{N}.pt`
- Best model saved to `weights/best.pt`
- Last model saved to `weights/last.pt`

---

## What Happens When Training Completes

The training script will automatically:

1. **Save Best Weights**
   - Location: `models/weights/pothole_yolov8n_best.pt`
   - This is the model you'll use for detection

2. **Generate Training Plots**
   - Loss curves
   - mAP curves
   - Precision-Recall curves
   - Confusion matrix

3. **Display Final Metrics**
   - mAP@0.5
   - mAP@0.5:0.95
   - Final loss values

---

## After Training: Next Steps

### 1. Test the Model on Dataset
```bash
python pothole_detector.py --test --model models/weights/pothole_yolov8n_best.pt
```

This will:
- Process images from your dataset
- Generate annotated images with bounding boxes
- Create segmentation masks
- Save results to `results/detections/`

### 2. Run Live Detection
```bash
python pothole_detector.py --model models/weights/pothole_yolov8n_best.pt
```

This will:
- Open your webcam
- Detect potholes in real-time
- Show severity classification (LOW, MEDIUM, HIGH)
- Track potholes across frames

### 3. Review Training Results
Check the training directory for:
- **results.csv**: Epoch-by-epoch metrics
- **results.png**: Training curves visualization
- **confusion_matrix.png**: Classification performance
- **labels.jpg**: Dataset label distribution
- **train_batch*.jpg**: Sample training batches

---

## Troubleshooting

### Training Interrupted?
If training stops unexpectedly, you can resume:
```bash
python scripts/train.py --model yolov8n --epochs 100 --batch 16 --resume
```

### Out of Memory?
Reduce batch size:
```bash
python scripts/train.py --model yolov8n --epochs 100 --batch 8
```

### Want Faster Training?
Use a smaller model or fewer epochs:
```bash
python scripts/train.py --model yolov8n --epochs 50 --batch 16
```

### Want Better Accuracy?
Use a larger model (requires more time and memory):
```bash
python scripts/train.py --model yolov8s --epochs 100 --batch 8
```

---

## Understanding the Metrics

### Loss Values (Lower is Better)
- **box_loss**: How well the model predicts bounding box locations
- **cls_loss**: How well the model classifies objects (pothole vs background)
- **dfl_loss**: Distribution focal loss for box regression

### mAP Scores (Higher is Better)
- **mAP@0.5**: Mean Average Precision at 50% IoU threshold
  - Good: > 0.7
  - Excellent: > 0.85
- **mAP@0.5:0.95**: Average mAP across IoU thresholds 0.5 to 0.95
  - Good: > 0.5
  - Excellent: > 0.7

---

## Dataset Information

**Training Set**: 590 images
- 2 corrupt GIF files were automatically skipped
- 3 corrupt JPEG files were automatically restored

**Validation Set**: 221 images

**Classes**: 1 (pothole)

**Dataset Location**: `Datasets/`
```
Datasets/
├── train/
│   ├── images/ (590 files)
│   └── labels/ (590 files)
├── val/
│   ├── images/ (221 files)
│   └── labels/ (221 files)
└── pothole_dataset.yaml
```

---

## Tips for Best Results

1. **Let it Train**: Don't interrupt the training. The model improves over time.

2. **Monitor Progress**: Use the monitor script to check progress without interrupting training.

3. **Check Plots**: After 10-20 epochs, review the plots to see if training is progressing well.

4. **Early Stopping**: The training has patience=50, so it will stop if no improvement for 50 epochs.

5. **Test Thoroughly**: After training, test on various images to ensure good performance.

---

## File Structure After Training

```
Real-time-pothole-detection/
├── models/
│   ├── weights/
│   │   └── pothole_yolov8n_best.pt  ← Your trained model
│   └── yolo_training/
│       └── pothole_yolov8n_20260209_151442/
│           ├── weights/
│           │   ├── best.pt
│           │   ├── last.pt
│           │   └── epoch*.pt
│           ├── results.csv
│           ├── results.png
│           ├── confusion_matrix.png
│           └── ...
├── logs/
│   └── training_20260209_151442.log
└── ...
```

---

## Questions?

- **How long will it take?** ~75-85 minutes on CPU
- **Can I use my computer while training?** Yes, but it may slow down
- **Will it use my GPU?** Only if you have CUDA-enabled GPU and PyTorch with CUDA
- **Can I stop and resume?** Yes, use `--resume` flag
- **What if accuracy is low?** Try more epochs, larger model, or more training data

---

**Happy Training! 🚀**
