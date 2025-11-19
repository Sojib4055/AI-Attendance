import numpy as np
from core.video_reader import VideoReader
from db.models import Person, IrisTemplate

class EnrollmentService:
    def __init__(self, pipeline, db_session, frame_skip: int = 3):
        self.pipeline = pipeline
        self.db = db_session
        self.frame_skip = frame_skip

    def enroll_from_video(self, person_meta: dict, video_path: str):
        person = Person(**person_meta)
        self.db.add(person)
        self.db.commit()

        reader = VideoReader(frame_skip=self.frame_skip)
        embeddings = []

        for _, frame in reader.iter_frames(video_path):
            emb_data = self.pipeline.process_frame(frame)
            for item in emb_data:
                embeddings.append(item["embedding"])

        if len(embeddings) == 0:
            raise RuntimeError("No iris embeddings extracted from enrollment video")

        avg_emb = np.mean(np.stack(embeddings, axis=0), axis=0)

        template = IrisTemplate(
            person_id=person.id,
            embedding=avg_emb.tobytes(),
            eye_side="unknown",
            quality_score=1.0,
        )
        self.db.add(template)
        self.db.commit()
        return person.id
