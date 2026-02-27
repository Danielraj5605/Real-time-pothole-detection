from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import math

from .database import engine, Base, get_db
from .models import Issue, IssueCreate, IssueResponse

# Initialize Database Tables (auto-create if not exist)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Civic Issue Reporting API")

# CORS Setup - Allow All for Development (Localhost + PWA)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility: Haversine Formula ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# --- Endpoints ---

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "pothole-backend"}

@app.post("/api/issues", response_model=IssueResponse)
def create_issue(issue: IssueCreate, db: Session = Depends(get_db)):
    """Report a new issue (e.g., pothole detected)"""
    new_issue = Issue(
        type=issue.type,
        latitude=issue.latitude,
        longitude=issue.longitude,
        severity=issue.severity,
        confidence=issue.confidence,
        description=issue.description,
        metadata_info=issue.metadata_info
    )
    db.add(new_issue)
    db.commit()
    db.refresh(new_issue)
    return new_issue

@app.get("/api/issues/nearby", response_model=List[IssueResponse])
def get_nearby_issues(lat: float, lon: float, radius: float = 100.0, db: Session = Depends(get_db)):
    """Find issues within `radius` meters of (lat, lon)"""
    # Note: In production with PostGIS, this would be a SQL query.
    # For now (proto), we fetch all and filter in Python (O(N)).
    # Acceptable for < 1000 records.
    
    all_issues = db.query(Issue).all()
    nearby = []
    
    for issue in all_issues:
        dist = haversine_distance(lat, lon, issue.latitude, issue.longitude)
        if dist <= radius:
            # We could add 'distance' to response if we updated the Pydantic model
            nearby.append(issue)
            
    return nearby
