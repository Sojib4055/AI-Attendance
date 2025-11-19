import argparse
from config.config_loader import load_config
from core.pipeline import IrisPipeline
from services.attendance_service import AttendanceService
from db.db_utils import get_session, init_db

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera_id", required=True)
    args = parser.parse_args()

    cfg = load_config()
    init_db(cfg["db_url"])
    SessionLocal = get_session(cfg)
    session = SessionLocal()

    pipeline = IrisPipeline(cfg)
    service = AttendanceService(
        pipeline,
        session,
        threshold=cfg["match"]["threshold"],
        frame_skip=cfg["video"]["frame_skip"],
    )
    service.process_video(args.video, args.camera_id)
    print("Attendance processing completed.")

if __name__ == "__main__":
    main()
