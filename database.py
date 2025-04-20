from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Create database engine
engine = create_engine('sqlite:///plant_identification.db', echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class IdentificationHistory(Base):
    __tablename__ = 'identification_history'

    id = Column(Integer, primary_key=True)
    plant_name = Column(String(50))
    confidence_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    image_data = Column(LargeBinary)  # Store the actual image data

    def __repr__(self):
        return f"<Identification(plant='{self.plant_name}', confidence={self.confidence_score}%)>"

# Create all tables
Base.metadata.create_all(engine)

def add_identification(plant_name: str, confidence: float, image_data: bytes):
    """Add a new identification record with image to the database"""
    session = Session()
    try:
        new_record = IdentificationHistory(
            plant_name=plant_name,
            confidence_score=confidence,
            image_data=image_data
        )
        session.add(new_record)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Error adding record: {e}")
        return False
    finally:
        session.close()

def get_recent_identifications(limit: int = 5):
    """Get recent identification history including images"""
    session = Session()
    try:
        records = session.query(IdentificationHistory)\
            .order_by(IdentificationHistory.timestamp.desc())\
            .limit(limit)\
            .all()
        return records
    finally:
        session.close()

def get_identification_image(id: int) -> bytes:
    """Retrieve image data for a specific identification"""
    session = Session()
    try:
        record = session.query(IdentificationHistory).get(id)
        return record.image_data if record else None
    finally:
        session.close()