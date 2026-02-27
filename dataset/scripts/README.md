# Pothole Detection Dataset Collection Scripts

This directory contains Python scripts for collecting synchronized multi-sensor data for pothole detection model training.

## Hardware Requirements

- **Raspberry Pi Zero WH** with Raspberry Pi OS
- **OV5647 Camera** (CSI connection)
- **MPU9250** 9-axis IMU (I2C connection)
- **NEO-6M GPS** module (UART connection) - **OPTIONAL**

## Quick Setup on Raspberry Pi

### 1. Wire the Hardware

```
MPU9250 (Required):
  VCC → Pin 1 (3.3V)
  GND → Pin 6 (Ground)
  SDA → Pin 3 (GPIO2)
  SCL → Pin 5 (GPIO3)   

NEO-6M GPS (Optional - NEO-6M-0-001):
  VCC → Pin 4 (5V) [recommended if module has regulator]
        OR Pin 1 (3.3V) [if module gets hot on 5V]
  GND → Pin 6 (Ground) [same ground as MPU9250]
  TX  → Pin 10 (GPIO15 RXD) [GPS transmit → Pi receive]
  RX  → Pin 8 (GPIO14 TXD) [GPS receive → Pi transmit]

Note: GPS TX/RX are 3.3V logic - safe for Pi GPIO!
```

### 2. Configure the Pi

```bash
# SSH into the Pi
ssh raspberrypi@10.91.79.190

# Enable I2C
sudo raspi-config
# Go to: Interface Options → I2C → Yes

# Configure UART for GPS
sudo raspi-config
# Go to: Interface Options → Serial Port
#   Login shell over serial: NO
#   Serial port hardware: YES

# Add to /boot/firmware/config.txt:
echo "enable_uart=1" | sudo tee -a /boot/firmware/config.txt

# Reboot
sudo reboot
```

### 3. Install Dependencies

```bash
# Create project directory
mkdir -p ~/pothole_dataset/data
cd ~/pothole_dataset

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
sudo apt update
sudo apt install -y i2c-tools
pip install smbus2 mpu9250-jmdev pyserial pynmea2

# Verify I2C (should show 0x68)
i2cdetect -y 1
```

### 4. Copy Scripts to Pi

From your laptop:
```bash
scp mpu9250_driver.py gps_driver.py collect_data.py raspberrypi@10.91.79.190:~/pothole_dataset/
```

## Usage

### Test Sensors Individually

```bash
cd ~/pothole_dataset
source venv/bin/activate

# Test MPU9250
python mpu9250_driver.py

# Test GPS
python gps_driver.py
```

### Collect Dataset

```bash
cd ~/pothole_dataset
source venv/bin/activate

# Start data collection (NO GPS - use this if GPS module is unavailable)
python collect_data.py --no-gps

# With options:
python collect_data.py --no-gps --calibrate    # Calibrate accelerometer first
python collect_data.py --no-gps --duration 300 # Record for 5 minutes
python collect_data.py --no-gps --no-video     # Accelerometer data only

# When GPS module is available:
python collect_data.py                         # With GPS enabled
```

### During Collection

- **[SPACE]** or **[ENTER]** - Mark a pothole when you encounter one
- **[Ctrl+C]** - Stop recording

### Copy Data to Laptop

From your laptop:
```bash
scp -r raspberrypi@10.91.79.190:~/pothole_dataset/data/session_* D:\projects\Realtime-Pothole-alert\dataset\
```

## Real-Time Accelerometer Overlay

For **low-latency** live display of accelerometer data on camera frames:

```bash
cd ~/pothole_dataset
source venv/bin/activate

# Install OpenCV (one-time)
pip install opencv-python-headless

# Basic usage (requires display or X11 forwarding)
python realtime_accel_overlay.py

# SSH without display (headless mode, print stats only)
python realtime_accel_overlay.py --headless --debug

# With smoothing filter (α=0.2 for smooth, 0.5 for responsive)
python realtime_accel_overlay.py --filter 0.2 --debug

# Benchmark latency (100 frames)
python realtime_accel_overlay.py --benchmark
```

**Key flags:**
- `--debug` - Show FPS and latency in real-time
- `--headless` - No display window (for SSH)
- `--filter 0.2` - Apply IIR low-pass filter
- `--sample-rate 200` - Accelerometer Hz (default: 200)

## Post-Processing (on Laptop)

Extract frames and match with sensor data:

```bash
cd D:\projects\Realtime-Pothole-alert\dataset\scripts

# Extract frames at 100ms intervals
python extract_frames.py ..\session_20260131_143000

# Extract frames only around labeled events
python extract_frames.py ..\session_20260131_143000 --around-labels
```

## Collected Data Format

Each session creates a folder with:

```
session_YYYYMMDD_HHMMSS/
├── metadata.json           # Session info and config
├── video/
│   └── raw_capture.h264    # Continuous video recording
├── accelerometer/
│   └── accel_log.csv       # Timestamped accelerometer data (100Hz)
├── gps/
│   └── gps_log.csv         # Timestamped GPS data (1Hz)
└── labels/
    └── events.json         # Manual pothole labels with timestamps
```

## Troubleshooting

### MPU9250 not detected
- Check wiring (SDA to Pin 3, SCL to Pin 5)
- Run `i2cdetect -y 1` to verify device at 0x68 or 0x69
- Make sure I2C is enabled in raspi-config

### GPS no fix
- Go outside with clear sky view
- Wait up to 2 minutes for first fix
- Check antenna connection

### Video not recording
- Verify camera works: `rpicam-still -o test.jpg`
- Check CSI cable connection
- Enable camera in raspi-config if needed





to run

in ssh


 cd ~/pothole_dataset && source venv/bin/activate

 python collect_data.py --no-gps


 in powershell to save


python process_session.py latest --open            
 