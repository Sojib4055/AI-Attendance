import cv2

from .asset_manager import ensure_eye_cascade

class IrisDetector:
    """IrisDetector wrapper.

    NOTE: This is a placeholder implementation using OpenCV's eye cascade
    so the pipeline can run end-to-end. For production, replace this logic
    with a proper IriTrack-based eye/iris localization model.
    """

    def __init__(self, cascade_path: str, min_size_ratio: float = 0.05):
        cascade_file = ensure_eye_cascade(cascade_path)
        self.cascade = cv2.CascadeClassifier(str(cascade_file))
        self.min_size_ratio = min_size_ratio

    def detect_eyes(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        min_size = int(min(h, w) * self.min_size_ratio)
        eyes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_size, min_size)
        )
        results = []
        for (x, y, ew, eh) in eyes:
            eye_crop = frame_bgr[y:y+eh, x:x+ew].copy()
            results.append(
                {
                    "eye_crop": eye_crop,
                    "bbox": (int(x), int(y), int(x+ew), int(y+eh)),
                    "confidence": 1.0,  # cascade has no confidence, so use 1.0
                }
            )
        return results
