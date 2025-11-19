import argparse
from config.config_loader import load_config
from core.pipeline import IrisPipeline
from services.enrollment_service import EnrollmentService
from db.db_utils import get_session, init_db

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--employee_code", required=True)
    parser.add_argument("--department", default="")
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    cfg = load_config()
    # init DB (creates tables if not exist)
    init_db(cfg["db_url"])
    SessionLocal = get_session(cfg)
    session = SessionLocal()

    pipeline = IrisPipeline(cfg)
    service = EnrollmentService(pipeline, session)

    person_id = service.enroll_from_video(
        {
            "name": args.name,
            "employee_code": args.employee_code,
            "department": args.department,
        },
        args.video,
    )
    print(f"Enrolled person_id={person_id}")

if __name__ == "__main__":
    main()
