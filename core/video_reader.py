import cv2

class VideoReader:
    def __init__(self, frame_skip: int = 5):
        self.frame_skip = frame_skip

    def iter_frames(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_skip == 0:
                yield frame_idx, frame
            frame_idx += 1
        cap.release()
