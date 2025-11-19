import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config.config_loader import load_config
from core.pipeline import IrisPipeline
from db.db_utils import get_session, init_db
from db.models import AttendanceEvent, IrisTemplate, Person
from services.attendance_service import AttendanceService
from services.enrollment_service import EnrollmentService

app = FastAPI(title="Iris Attendance API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CFG = load_config()
init_db(CFG["db_url"])
SessionFactory = get_session(CFG)
PIPELINE = IrisPipeline(CFG)
VIDEO_CFG = CFG.get("video", {})
MATCH_CFG = CFG.get("match", {})


def get_db() -> Session:
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()


async def save_upload(upload: UploadFile, prefix: str) -> str:
    suffix = Path(upload.filename or "video").suffix or ".mp4"
    fd, tmp_path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as buffer:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
    except Exception:
        os.remove(tmp_path)
        raise
    finally:
        await upload.close()
    return tmp_path


class EnrollmentResponse(BaseModel):
    person_id: int
    message: str


class AttendanceProcessResponse(BaseModel):
    events_logged: int
    message: str


class PersonResponse(BaseModel):
    id: int
    name: str
    employee_code: str
    department: Optional[str]
    created_at: datetime
    templates: int


class AttendanceEventResponse(BaseModel):
    id: int
    person_id: int
    camera_id: str
    video_path: str
    timestamp: datetime
    score: float
    frame_idx: int


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/enroll", response_model=EnrollmentResponse)
async def enroll_person(
    name: str = Form(...),
    employee_code: str = Form(...),
    department: str = Form(""),
    video: UploadFile = File(...),
    session: Session = Depends(get_db),
):
    tmp_path = await save_upload(video, "enroll")
    person_id = None
    try:
        service = EnrollmentService(
            PIPELINE,
            session,
            frame_skip=VIDEO_CFG.get("frame_skip", 3),
        )
        person_id = service.enroll_from_video(
            {
                "name": name.strip(),
                "employee_code": employee_code.strip(),
                "department": department.strip(),
            },
            tmp_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return EnrollmentResponse(person_id=person_id, message="Enrollment completed")


@app.post("/attendance/process", response_model=AttendanceProcessResponse)
async def process_attendance(
    camera_id: str = Form(...),
    video: UploadFile = File(...),
    session: Session = Depends(get_db),
):
    tmp_path = await save_upload(video, "attendance")
    events_logged = 0
    try:
        before_count = session.query(AttendanceEvent).count()
        service = AttendanceService(
            PIPELINE,
            session,
            threshold=MATCH_CFG.get("threshold", 0.7),
            frame_skip=VIDEO_CFG.get("frame_skip", 5),
        )
        service.process_video(tmp_path, camera_id.strip())
        after_count = session.query(AttendanceEvent).count()
        events_logged = max(after_count - before_count, 0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return AttendanceProcessResponse(
        events_logged=events_logged,
        message="Attendance processing completed",
    )


@app.get("/persons", response_model=List[PersonResponse])
def list_persons(session: Session = Depends(get_db)):
    people = session.query(Person).order_by(Person.created_at.desc()).all()
    return [
        PersonResponse(
            id=p.id,
            name=p.name,
            employee_code=p.employee_code,
            department=p.department,
            created_at=p.created_at,
            templates=len(p.iris_templates),
        )
        for p in people
    ]


@app.get("/attendance/events", response_model=List[AttendanceEventResponse])
def list_attendance_events(limit: int = 50, session: Session = Depends(get_db)):
    limit = max(1, min(limit, 200))
    events = (
        session.query(AttendanceEvent)
        .order_by(AttendanceEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        AttendanceEventResponse(
            id=e.id,
            person_id=e.person_id,
            camera_id=e.camera_id,
            video_path=e.video_path,
            timestamp=e.timestamp,
            score=float(e.score),
            frame_idx=e.frame_idx,
        )
        for e in events
    ]


@app.get("/templates", response_model=List[int])
def list_template_ids(session: Session = Depends(get_db)):
    templates = session.query(IrisTemplate.id).all()
    return [t[0] for t in templates]
