import cv2
import numpy as np

class IrisSegmenter:
    """Placeholder iris segmenter.

    This approximates the iris by running a Hough circle transform on the
    grayscale eye crop. For real deployment, replace with a trained RITnet
    model and return its binary mask.
    """

    def __init__(self):
        pass

    def segment(self, eye_crop_bgr):
        gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        h, w = gray.shape[:2]

        # HoughCircles to guess iris region
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=min(h, w) // 4,
            param1=100,
            param2=30,
            minRadius=min(h, w) // 6,
            maxRadius=min(h, w) // 2,
        )

        mask = np.zeros_like(gray, dtype=np.uint8)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # take the largest circle as iris
            c = max(circles[0, :], key=lambda x: x[2])
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            cv2.circle(mask, (cx, cy), r, 1, thickness=-1)
        return mask
