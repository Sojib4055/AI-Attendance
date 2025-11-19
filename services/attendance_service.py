import numpy as np
from datetime import datetime
from core.video_reader import VideoReader
from core.matcher import IrisMatcher
from db.models import IrisTemplate, AttendanceEvent

class AttendanceService:
    def __init__(self, pipeline, db_session, threshold: float = 0.7, frame_skip: int = 5):
        self.pipeline = pipeline
        self.db = db_session
        self.matcher = IrisMatcher(threshold=threshold)
        self.frame_skip = frame_skip

    def _load_templates(self):
        templates = []
        for t in self.db.query(IrisTemplate).all():
            emb = np.frombuffer(t.embedding, dtype=np.float32)
            templates.append({"person_id": t.person_id, "embedding": emb})
        return templates

    def process_video(self, video_path: str, camera_id: str):
        templates = self._load_templates()
        reader = VideoReader(frame_skip=self.frame_skip)

        for frame_idx, frame in reader.iter_frames(video_path):
            emb_data = self.pipeline.process_frame(frame)
            for item in emb_data:
                person_id, score = self.matcher.match(item["embedding"], templates)
                if person_id is None:
                    continue
                event = AttendanceEvent(
                    person_id=person_id,
                    camera_id=camera_id,
                    video_path=video_path,
                    timestamp=datetime.utcnow(),
                    score=score,
                    frame_idx=frame_idx,
                )
                self.db.add(event)
        self.db.commit()
