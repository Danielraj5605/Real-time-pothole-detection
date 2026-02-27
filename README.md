# Real-Time Pothole Detection & Alert System (Prototype)

This is a minor academic project prototype for a civic infrastructure reporting platform. It includes a backend server, a PWA frontend, and scripts to simulate data upload.

## 📁 Project Structure

*   `server/`: FastAPI backend (API & Database).
*   `webapp/`: Frontend PWA (Leaflet Map + Geolocation).
*   `scripts/`: Utilities (Mock Data Upload).
*   `detection/`: (External) Offline YOLO processing.

## 🚀 Setup & Run

### 1. Prerequisites
*   Python 3.9+
*   PostgreSQL (or Neon Serverless account)

### 2. Install Dependencies
```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary requests pydantic
```

### 3. Configure Database
Set your connection string. For local testing with SQLite (simplest), no change needed (it defaults to `potholes.db` if no env var is set).
For **Neon Postgres**:
```bash
# Windows (PowerShell)
$env:DATABASE_URL="postgresql://user:password@endpoint.neon.tech/dbname?sslmode=require"
```

### 4. Run the Backend
```bash
uvicorn server.main:app --reload
```
API will be running at: `http://127.0.0.1:8000`
Docs at: `http://127.0.0.1:8000/docs`

### 5. Run the Frontend (React PWA)
Navigate to the React folder and start the dev server:
```bash
cd webapp-react
npm run dev
```
Open the URL shown (usually `http://localhost:5173`).
*   **Important:** Allow Geolocation when asked.


### 6. Upload Mock Data
To simulate detections coming from the "edge device" (Python script):
```bash
python scripts/upload_detections.py
```
This will upload 10 random potholes along a generated trajectory.

## 🧪 Testing
1.  Open the **Frontend** (`index.html`). Allow GPS access.
2.  Run the **Backend**.
3.  Run the **Upload Script**.
4.  Watch markers appear on the map!
5.  If you are "close" to a mock pothole (use Chrome DevTools Sensors to fake your location), an **ALERT** will pop up.
