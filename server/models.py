from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from .database import Base
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

# --- SQLAlchemy Models (Database Table) ---

class Issue(Base):
    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(50), nullable=False)  # 'pothole', 'garbage', etc.
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    severity = Column(String(20)) # 'LOW', 'MEDIUM', 'HIGH'
    confidence = Column(Float)
    status = Column(String(20), default="REPORTED") # 'REPORTED', 'VERIFIED', 'FIXED'
    
    image_url = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    metadata_info = Column(JSON, nullable=True) # Renamed from metadata to avoid conflict
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# --- Pydantic Schemas (API Input/Output) ---

class IssueCreate(BaseModel):
    type: str # 'pothole'
    latitude: float
    longitude: float
    severity: Optional[str] = None
    confidence: Optional[float] = None
    description: Optional[str] = None
    metadata_info: Optional[Dict[str, Any]] = None

class IssueResponse(IssueCreate):
    id: int
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True
