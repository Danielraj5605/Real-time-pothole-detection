import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Neon Connection String
DATABASE_URL = "postgresql://neondb_owner:npg_VHT3ek9LRXtM@ep-snowy-glitter-aivj2mac-pooler.c-4.us-east-1.aws.neon.tech/neondb?sslmode=require"

# SQLAlchemy Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
