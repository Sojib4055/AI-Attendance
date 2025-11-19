from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

def get_engine(db_url: str):
    engine = create_engine(db_url, echo=False, future=True)
    return engine

def init_db(db_url: str):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine

def get_session_from_url(db_url: str):
    engine = get_engine(db_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal

def get_session(cfg):
    db_url = cfg.get("db_url", "sqlite:///iris_attendance.db")
    engine = get_engine(db_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal
