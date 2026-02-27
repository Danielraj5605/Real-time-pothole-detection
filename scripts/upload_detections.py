import requests
import time
import random
import math

# Configuration
API_URL = "http://127.0.0.1:8000/api/issues"
NUM_POTHOLES = 10
DELAY_SECONDS = 0.5

# Mock GPS Trajectory Generator (Simulating a drive down a road)
# Location: Near user's current position, Guduvancheri
START_LAT = 12.85460
START_LON = 80.06801

def generate_mock_trajectory(start_lat, start_lon, steps):
    path = []
    lat, lon = start_lat, start_lon
    
    for i in range(steps):
        # Move roughly Northeast
        lat += 0.0001 + random.uniform(-0.00002, 0.00002)
        lon += 0.0001 + random.uniform(-0.00002, 0.00002)
        path.append((lat, lon))
    return path

def upload_mock_data():
    print(f"--- Starting Mock Pothole Upload to {API_URL} ---")
    
    # 1. Generate a path
    trajectory = generate_mock_trajectory(START_LAT, START_LON, NUM_POTHOLES)
    
    for i, (lat, lon) in enumerate(trajectory):
        # 2. Create a Mock Pothole Detection
        severity = random.choice(["LOW", "MEDIUM", "HIGH"])
        confidence = round(random.uniform(0.5, 0.99), 2)
        
        payload = {
            "type": "pothole",
            "latitude": lat,
            "longitude": lon,
            "severity": severity,
            "confidence": confidence,
            "description": f"Mock pothole #{i+1} detected on test drive.",
            "metadata_info": {
                "source": "simulation_script",
                "area_ratio": round(random.uniform(0.01, 0.15), 3)
            }
        }
        
        try:
            # 3. Upload to Backend
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                print(f"[SUCCESS] Uploaded Pothole #{data['id']} at ({lat:.4f}, {lon:.4f}) Severity: {severity}")
            else:
                print(f"[ERROR] Failed to upload: {response.status_code} - {response.text}")
        except Exception as e:
             print(f"[EXCEPTION] Connection failed: {e}")
             print("Is the server running? (uvicorn server.main:app --reload)")
             break
        
        time.sleep(DELAY_SECONDS)

    print("--- Upload Complete ---")

if __name__ == "__main__":
    upload_mock_data()
