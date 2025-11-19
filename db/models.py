from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Float,
    LargeBinary,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    employee_code = Column(String(100), unique=True, nullable=False)
    department = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    iris_templates = relationship("IrisTemplate", back_populates="person")
    attendance_events = relationship("AttendanceEvent", back_populates="person")

    def __repr__(self):
        return f"<Person id={self.id} name={self.name} employee_code={self.employee_code}>"

class IrisTemplate(Base):
    __tablename__ = "iris_templates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    eye_side = Column(String(20), default="unknown", nullable=False)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    person = relationship("Person", back_populates="iris_templates")

    def __repr__(self):
        return f"<IrisTemplate id={self.id} person_id={self.person_id} eye_side={self.eye_side}>"

class AttendanceEvent(Base):
    __tablename__ = "attendance_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False)
    camera_id = Column(String(100), nullable=False)
    video_path = Column(String(1024), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    score = Column(Float, nullable=False)
    frame_idx = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    person = relationship("Person", back_populates="attendance_events")

    def __repr__(self):
        return (
            f"<AttendanceEvent id={self.id} person_id={self.person_id} "
            f"camera_id={self.camera_id} frame_idx={self.frame_idx} score={self.score:.3f}>"
        )
