#Previous :

# from sqlalchemy import Column, Integer, String, Float, DateTime
# from sqlalchemy.sql import func
# from .database import Base

# class PredictionResult(Base):
#     __tablename__ = "prediction_results"

#     id = Column(Integer, primary_key=True, index=True)
#     filename = Column(String, index=True)
#     predicted_class = Column(String, index=True)
#     confidence = Column(Float)
#     timestamp = Column(DateTime(timezone=True), server_default=func.now())

#Updated :

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql.expression import text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from api.database import Base

class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    predicted_class = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    owner = relationship("User")

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, nullable=False)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=text('now()'))