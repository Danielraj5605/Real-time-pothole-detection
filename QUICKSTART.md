# Quick Start Guide - Pothole Detection Training

## Step-by-Step Instructions

### ⚡ Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.8 or higher installed
- [ ] At least 8GB RAM
- [ ] 20GB free disk space
- [ ] (Optional) NVIDIA GPU with CUDA support

---

## 🚀 Getting Started (5 Minutes)

### Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd "d:\Personal Projects\Real-time-pothole-detection"

# Install required packages
pip install -r requirements.txt
```

**Expected output:** All packages installed successfully

---

### Step 2: Validate Your Dataset (1 minute)

```bash
# Check if labels exist and are properly formatted
python prepare_labels.py --mode validate
```

**What to expect:**
- Total images count
- Images with/without labels
- Class distribution (low, medium, high)

**⚠️ IMPORTANT:** If you see "0 images with labels", you need to create annotations first!

---

### Step 3: Create Labels (If Needed)

**If you don't have labels yet:**

1. **Download LabelImg:**
   ```bash
   pip install labelImg
   labelImg
   ```

2. **Annotation Instructions:**
   - Open `Datasets/images` folder
   - Draw boxes around potholes
   - Assign severity class:
     - **Class 0**: Low severity (small, shallow)
     - **Class 1**: Medium severity (moderate damage)
     - **Class 2**: High severity (large, deep)
   - Save in YOLO format
   - Labels will be saved to `Datasets/labels`

3. **Quick annotation tips:**
   - Use keyboard shortcuts (W = draw box, D = next image)
   - Be consistent with severity classification
   - Aim for at least 100 annotated images to start

---

### Step 4: Visualize Your Data (1 minute)

```bash
# View some annotated samples
python prepare_labels.py --mode visualize
```

**This will show:**
- Images with bounding boxes
- Severity labels color-coded:
  - 🟢 Green = Low
  - 🟠 Orange = Medium
  - 🔴 Red = High

---

### Step 5: Start Training (1 minute to start)

```bash
# Begin model training
python train_yolov5.py
```

**The script will:**
1. ✅ Check your environment (GPU/CPU)
2. ✅ Prepare dataset splits (70% train, 20% val, 10% test)
3. ✅ Clone YOLOv5 repository (first time only)
4. ✅ Download pretrained weights
5. ✅ Start training

**When prompted "Start training? (yes/no):"** type `yes`

---

## ⏱️ Training Timeline

| Hardware | Expected Time |
|----------|---------------|
| RTX 3060/3070 | 2-3 hours |
| RTX 2060 | 4-5 hours |
| GTX 1060 | 6-8 hours |
| CPU only | 12-24 hours |

**💡 Tip:** Training will continue even if you close the terminal (runs in background)

---

## 📊 Monitoring Training

### During Training

Watch for these metrics in the terminal:
- **Epoch**: Current training iteration (0-100)
- **GPU Memory**: Should be 70-90% utilized
- **Loss**: Should decrease over time
- **mAP**: Should increase over time

### After Training

Check the results:
```bash
# Training results are saved in:
runs/pothole_detection_YYYYMMDD_HHMMSS/

# Key files:
# - weights/best.pt (best model)
# - weights/last.pt (last epoch)
# - results.png (training curves)
# - confusion_matrix.png (performance)
```

---

## 🧪 Testing Your Model

### Quick Test (30 seconds)

```bash
# Test on a single image
python test_model.py --mode single
```

### Full Evaluation (2-5 minutes)

```bash
# Evaluate on entire test set
python test_model.py --mode test_set
```

### Speed Benchmark

```bash
# Check inference speed
python test_model.py --mode benchmark
```

---

## 🎯 Expected Results

### Good Training Indicators:

✅ **mAP@0.5 > 0.80** (80% accuracy)  
✅ **Precision > 0.75**  
✅ **Recall > 0.70**  
✅ **Loss decreasing steadily**  

### If Results Are Poor:

❌ **mAP < 0.50** → Need more/better annotations  
❌ **High loss** → Train for more epochs  
❌ **Low recall** → Missing annotations in dataset  
❌ **Low precision** → Too many false positives  

---

## 🔧 Common Issues & Solutions

### Issue 1: "CUDA out of memory"

**Solution:**
```python
# Edit train_yolov5.py, line ~50
BATCH_SIZE = 8  # Reduce from 16 to 8 or 4
```

### Issue 2: "No labels found"

**Solution:**
```bash
# Validate label format
python prepare_labels.py --mode check --label_file Datasets/labels/sample.txt

# Check label location
# Labels must be in: Datasets/labels/
# Format: <class_id> <x_center> <y_center> <width> <height>
```

### Issue 3: Training is very slow

**Solutions:**
- Use smaller model: Change `MODEL_SIZE = "yolov5n"` in train_yolov5.py
- Reduce image size: Change `IMG_SIZE = 416` (from 640)
- Enable mixed precision training (automatic with GPU)

### Issue 4: "ImportError: No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📱 Next Steps After Training

### 1. Optimize for Raspberry Pi

The training script will ask if you want to optimize for edge deployment.
Say **yes** to create:
- `best.onnx` - Optimized for Raspberry Pi
- `best.torchscript` - Alternative format

### 2. Deploy to Raspberry Pi

```bash
# On your computer
scp weights/best.onnx pi@raspberrypi.local:~/

# On Raspberry Pi
python3 inference_pi.py --model best.onnx
```

### 3. Integrate Sensors

- Connect MPU6050 accelerometer
- Connect NEO-6M GPS module
- Implement sensor fusion
- Add real-time alerts

---

## 📋 Checklist

Before you start training, make sure:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset has at least 100 annotated images
- [ ] Labels validated (`python prepare_labels.py --mode validate`)
- [ ] At least 20GB free disk space
- [ ] GPU drivers installed (if using GPU)

---

## 🆘 Getting Help

### Check Training Logs

```bash
# View latest training log
cat runs/pothole_detection_*/train.log
```

### Validate Everything

```bash
# Run full validation
python prepare_labels.py --mode validate
```

### Test Environment

```python
# Test if PyTorch can see GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 💡 Pro Tips

1. **Start Small**: Begin with 100-200 images to test the pipeline
2. **Quality > Quantity**: Better to have 500 well-annotated images than 2000 poor ones
3. **Monitor GPU**: Use `nvidia-smi` to watch GPU utilization
4. **Save Checkpoints**: Training saves checkpoints every epoch
5. **Use Augmentation**: Enabled by default, helps with small datasets

---

## 🎓 Learning Resources

- [YOLOv5 Tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [Object Detection Basics](https://www.youtube.com/watch?v=yqkISICHH-U)
- [YOLO Format Explained](https://roboflow.com/formats/yolov5-pytorch-txt)

---

## ⏭️ What's Next?

After successful training:

1. ✅ Test model accuracy
2. ✅ Optimize for Raspberry Pi
3. ✅ Build complete detection system
4. ✅ Integrate GPS and accelerometer
5. ✅ Deploy and test in real-world

---

**Ready to start? Run:**

```bash
python train_yolov5.py
```

**Good luck! 🚀**
