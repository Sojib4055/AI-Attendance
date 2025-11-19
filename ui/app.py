import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import streamlit as st

from config.config_loader import load_config
from core.pipeline import IrisPipeline
from db.db_utils import get_session, init_db
from db.models import AttendanceEvent, IrisTemplate, Person
from services.attendance_service import AttendanceService
from services.enrollment_service import EnrollmentService

st.set_page_config(page_title="Iris Attendance UI", layout="wide")
st.title("Iris Attendance Control Panel")


def bootstrap():
    cfg = load_config()
    init_db(cfg["db_url"])
    SessionLocal = get_session(cfg)
    pipeline = IrisPipeline(cfg)
    return cfg, SessionLocal, pipeline


@st.cache_resource(show_spinner=False)
def get_dependencies():
    return bootstrap()


try:
    CFG, SessionFactory, PIPELINE = get_dependencies()
except Exception as exc:  # pragma: no cover - surfaced in UI
    st.error(f"Failed to bootstrap the pipeline: {exc}")
    st.stop()

VIDEO_CFG = CFG.get("video", {})
MATCH_CFG = CFG.get("match", {})


def save_uploaded_video(upload, prefix: str) -> str:
    suffix = Path(upload.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, prefix=prefix + "_", suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        return tmp.name


@contextmanager
def db_session_scope():
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()


def render_enrollment_page():
    st.subheader("Enroll a New Person")
    with st.form("enroll_form"):
        name = st.text_input("Full Name")
        employee_code = st.text_input("Employee Code")
        department = st.text_input("Department", value="")
        enrollment_video = st.file_uploader(
            "Enrollment Video", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False
        )
        submitted = st.form_submit_button("Enroll Person")

    if not submitted:
        return

    if not (name and employee_code and enrollment_video):
        st.warning("Provide name, employee code, and an enrollment video before enrolling.")
        return

    tmp_path: Optional[str] = None
    with st.spinner("Running enrollment pipeline…"):
        try:
            tmp_path = save_uploaded_video(enrollment_video, "enroll")
            with db_session_scope() as session:
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
            st.success(f"Enrollment completed. Assigned person ID: {person_id}")
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.error(f"Enrollment failed: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


def render_attendance_page():
    st.subheader("Process Attendance Video")
    with st.form("attendance_form"):
        camera_id = st.text_input("Camera ID")
        video_file = st.file_uploader(
            "Attendance Video", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False
        )
        submitted = st.form_submit_button("Process Video")

    if not submitted:
        return

    if not (camera_id and video_file):
        st.warning("Provide a camera ID and a video before processing.")
        return

    tmp_path: Optional[str] = None
    with st.spinner("Generating embeddings and logging attendance…"):
        try:
            tmp_path = save_uploaded_video(video_file, "attendance")
            with db_session_scope() as session:
                before_count = session.query(AttendanceEvent).count()
                service = AttendanceService(
                    PIPELINE,
                    session,
                    threshold=MATCH_CFG.get("threshold", 0.7),
                    frame_skip=VIDEO_CFG.get("frame_skip", 5),
                )
                service.process_video(tmp_path, camera_id.strip())
                after_count = session.query(AttendanceEvent).count()
            new_events = max(after_count - before_count, 0)
            st.success(f"Attendance processing complete. Logged {new_events} new event(s).")
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.error(f"Attendance processing failed: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)


def render_database_page():
    st.subheader("Database Overview")
    with db_session_scope() as session:
        total_people = session.query(Person).count()
        total_templates = session.query(IrisTemplate).count()
        total_events = session.query(AttendanceEvent).count()

        col1, col2, col3 = st.columns(3)
        col1.metric("Enrolled Persons", total_people)
        col2.metric("Stored Iris Templates", total_templates)
        col3.metric("Attendance Events", total_events)

        st.markdown("### Recent Attendance Events")
        events = (
            session.query(AttendanceEvent)
            .order_by(AttendanceEvent.timestamp.desc())
            .limit(25)
            .all()
        )
        if events:
            event_rows = [
                {
                    "Event ID": e.id,
                    "Person ID": e.person_id,
                    "Camera": e.camera_id,
                    "Score": round(float(e.score), 3),
                    "Frame": e.frame_idx,
                    "Timestamp (UTC)": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for e in events
            ]
            st.dataframe(event_rows, use_container_width=True)
        else:
            st.info("No attendance events recorded yet.")

        st.markdown("### Enrolled Persons")
        people = session.query(Person).order_by(Person.created_at.desc()).all()
        if people:
            person_rows = [
                {
                    "Person ID": p.id,
                    "Name": p.name,
                    "Employee Code": p.employee_code,
                    "Department": p.department or "",
                    "Created": p.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Templates": len(p.iris_templates),
                }
                for p in people
            ]
            st.dataframe(person_rows, use_container_width=True)
        else:
            st.info("No enrolled persons yet.")


PAGES = {
    "Enroll": render_enrollment_page,
    "Attendance": render_attendance_page,
    "Database": render_database_page,
}

selection = st.sidebar.radio("Navigation", list(PAGES.keys()), index=0)
PAGES[selection]()
