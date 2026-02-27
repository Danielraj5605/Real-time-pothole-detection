"""
Mock Data Generator for Pothole Detection System

Generates realistic sensor data (GPS, accelerometer, gyroscope) with simulated pothole events
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random
from pathlib import Path


class MockDataGenerator:
    """Generate realistic mock data for pothole detection testing"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_trip(self, 
                     duration_minutes=10,
                     num_potholes=5,
                     sample_rate_hz=50,
                     start_lat=40.4474,
                     start_lon=-79.9441):
        """
        Generate a complete trip with sensor data and potholes
        
        Args:
            duration_minutes: Trip duration
            num_potholes: Number of potholes to simulate
            sample_rate_hz: Accelerometer sampling rate
            start_lat: Starting latitude
            start_lon: Starting longitude
            
        Returns:
            sensor_df: DataFrame with sensor readings
            pothole_df: DataFrame with pothole events
        """
        
        # Calculate total samples
        total_samples = duration_minutes * 60 * sample_rate_hz
        
        # Generate timestamps
        start_time = datetime.now().timestamp()
        timestamps = [start_time + i/sample_rate_hz for i in range(total_samples)]
        
        # Generate GPS trajectory (simulate vehicle movement)
        latitudes, longitudes, speeds = self._generate_gps_trajectory(
            total_samples, start_lat, start_lon
        )
        
        # Generate baseline accelerometer data (normal driving)
        accel_x, accel_y, accel_z = self._generate_baseline_accel(total_samples)
        gyro_x, gyro_y, gyro_z = self._generate_baseline_gyro(total_samples)
        
        # Insert pothole events
        pothole_timestamps = []
        pothole_locations = []
        pothole_severities = []
        
        for i in range(num_potholes):
            # Random pothole location in trip
            pothole_idx = random.randint(
                int(total_samples * 0.1),  # Not at start
                int(total_samples * 0.9)   # Not at end
            )
            
            pothole_time = timestamps[pothole_idx]
            pothole_lat = latitudes[pothole_idx]
            pothole_lon = longitudes[pothole_idx]
            
            # Random severity
            severity = random.choice(['LOW', 'MEDIUM', 'HIGH'])
            
            # Inject pothole signature into accelerometer data
            accel_x, accel_y, accel_z = self._inject_pothole_signature(
                accel_x, accel_y, accel_z, pothole_idx, severity
            )
            
            pothole_timestamps.append(pothole_time)
            pothole_locations.append((pothole_lat, pothole_lon))
            pothole_severities.append(severity)
        
        # Create sensor dataframe
        sensor_df = pd.DataFrame({
            'timestamp': timestamps,
            'latitude': latitudes,
            'longitude': longitudes,
            'speed': speeds,
            'accelerometerX': accel_x,
            'accelerometerY': accel_y,
            'accelerometerZ': accel_z,
            'gyroX': gyro_x,
            'gyroY': gyro_y,
            'gyroZ': gyro_z
        })
        
        # Create pothole dataframe
        pothole_df = pd.DataFrame({
            'timestamp': pothole_timestamps,
            'latitude': [loc[0] for loc in pothole_locations],
            'longitude': [loc[1] for loc in pothole_locations],
            'severity': pothole_severities
        })
        
        # Sort by timestamp
        pothole_df = pothole_df.sort_values('timestamp').reset_index(drop=True)
        
        return sensor_df, pothole_df
    
    def _generate_gps_trajectory(self, num_samples, start_lat, start_lon):
        """Generate realistic GPS trajectory"""
        
        # Simulate vehicle moving at varying speeds
        speeds = []
        current_speed = 0.0
        
        for i in range(num_samples):
            # Gradual acceleration/deceleration
            if i < num_samples * 0.1:  # Accelerating
                current_speed = min(current_speed + 0.01, 15.0)
            elif i > num_samples * 0.9:  # Decelerating
                current_speed = max(current_speed - 0.01, 0.0)
            else:  # Cruising with variations
                current_speed += np.random.normal(0, 0.5)
                current_speed = np.clip(current_speed, 5.0, 15.0)
            
            speeds.append(current_speed)
        
        # Convert speed to GPS displacement
        # Approximate: 1 degree ≈ 111 km
        latitudes = [start_lat]
        longitudes = [start_lon]
        
        for i in range(1, num_samples):
            # Speed in m/s, sample rate in Hz
            distance_m = speeds[i] / 50.0  # meters per sample
            
            # Random direction (mostly forward with slight variations)
            angle = np.random.normal(0, 0.1)  # radians
            
            dlat = (distance_m * np.cos(angle)) / 111000  # degrees
            dlon = (distance_m * np.sin(angle)) / (111000 * np.cos(np.radians(latitudes[-1])))
            
            latitudes.append(latitudes[-1] + dlat)
            longitudes.append(longitudes[-1] + dlon)
        
        return latitudes, longitudes, speeds
    
    def _generate_baseline_accel(self, num_samples):
        """Generate baseline accelerometer data (normal driving)"""
        
        # Normal driving vibrations
        accel_x = np.random.normal(0.05, 0.03, num_samples)
        accel_y = np.random.normal(-0.96, 0.05, num_samples)  # Gravity
        accel_z = np.random.normal(0.20, 0.05, num_samples)
        
        # Add road texture noise
        accel_x += np.random.normal(0, 0.01, num_samples)
        accel_y += np.random.normal(0, 0.01, num_samples)
        accel_z += np.random.normal(0, 0.01, num_samples)
        
        return accel_x, accel_y, accel_z
    
    def _generate_baseline_gyro(self, num_samples):
        """Generate baseline gyroscope data"""
        
        gyro_x = np.random.normal(0, 0.02, num_samples)
        gyro_y = np.random.normal(0, 0.02, num_samples)
        gyro_z = np.random.normal(0, 0.01, num_samples)
        
        return gyro_x, gyro_y, gyro_z
    
    def _inject_pothole_signature(self, accel_x, accel_y, accel_z, 
                                  pothole_idx, severity):
        """Inject pothole impact signature into accelerometer data"""
        
        # Pothole impact parameters based on severity
        impact_params = {
            'LOW': {'peak': 0.5, 'duration': 10, 'rms': 0.15},
            'MEDIUM': {'peak': 1.5, 'duration': 15, 'rms': 0.5},
            'HIGH': {'peak': 3.0, 'duration': 20, 'rms': 1.0}
        }
        
        params = impact_params[severity]
        
        # Create impact signature (sharp spike + decay)
        duration = params['duration']
        half_dur = duration // 2
        
        for i in range(duration):
            idx = pothole_idx + i - half_dur
            if 0 <= idx < len(accel_x):
                # Gaussian-like impact
                t = (i - half_dur) / half_dur
                impact = params['peak'] * np.exp(-t**2)
                
                accel_y[idx] += impact * np.random.uniform(0.8, 1.2)
                accel_z[idx] += impact * 0.5 * np.random.uniform(0.8, 1.2)
                accel_x[idx] += impact * 0.3 * np.random.uniform(-1, 1)
        
        return accel_x, accel_y, accel_z


def main():
    """Generate mock trip data"""
    
    print("=" * 60)
    print("  MOCK DATA GENERATOR - Pothole Detection System")
    print("=" * 60)
    
    generator = MockDataGenerator()
    
    # Create output directory
    output_dir = Path('Datasets/Pothole')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate 5 trips
    for trip_num in range(1, 6):
        print(f"\n[{trip_num}/5] Generating mock trip {trip_num}...")
        
        sensor_df, pothole_df = generator.generate_trip(
            duration_minutes=random.randint(5, 15),
            num_potholes=random.randint(3, 8),
            sample_rate_hz=50
        )
        
        # Save to CSV
        sensor_path = output_dir / f'mock_trip{trip_num}_sensors.csv'
        pothole_path = output_dir / f'mock_trip{trip_num}_potholes.csv'
        
        sensor_df.to_csv(sensor_path, index=False)
        pothole_df.to_csv(pothole_path, index=False)
        
        print(f"  ✅ Sensor samples: {len(sensor_df)}")
        print(f"  ✅ Potholes: {len(pothole_df)}")
        
        severity_counts = pothole_df['severity'].value_counts()
        print(f"  ✅ Severities: {severity_counts.to_dict()}")
        print(f"  📁 Saved to: {sensor_path.name}, {pothole_path.name}")
    
    print("\n" + "=" * 60)
    print("  ✅ MOCK DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\n📂 Output directory: {output_dir.absolute()}")
    print(f"📊 Generated 5 trips with sensor data and pothole events")
    print(f"\nNext steps:")
    print(f"  1. Run: python scripts/test_accelerometer.py")
    print(f"  2. Run: python scripts/end_to_end_test.py")


if __name__ == "__main__":
    main()
