import numpy as np
import cv2

class DaugmanNormalizer:
    def __init__(self, radial_res: int = 64, angular_res: int = 512):
        self.radial_res = radial_res
        self.angular_res = angular_res

    def normalize(self, eye_crop, iris_mask, pupil_center, pupil_radius, iris_radius):
        """Rubber-sheet normalization.

        eye_crop: BGR eye crop
        iris_mask: uint8 mask with 1 for iris region
        pupil_center: (cx, cy)
        pupil_radius: int
        iris_radius: int
        Returns: normalized iris strip [radial_res, angular_res] in float32.
        """
        cx, cy = pupil_center
        H, W = iris_mask.shape[:2]
        theta = np.linspace(0, 2 * np.pi, self.angular_res, endpoint=False)
        r = np.linspace(0, 1, self.radial_res)

        # radii from pupil to iris
        r_mat = pupil_radius + r[:, None] * (iris_radius - pupil_radius)
        theta_mat = theta[None, :]

        x = cx + r_mat * np.cos(theta_mat)
        y = cy + r_mat * np.sin(theta_mat)

        # clip coordinates
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

        # sample from grayscale version
        gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # bilinear sampling
        x0 = np.floor(x).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y0 = np.floor(y).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, H - 1)

        Ia = gray[y0, x0]
        Ib = gray[y0, x1]
        Ic = gray[y1, x0]
        Id = gray[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        norm = wa * Ia + wb * Ib + wc * Ic + wd * Id

        # normalize to [0,1]
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6)
        return norm.astype(np.float32)

    def estimate_geometry_from_mask(self, iris_mask):
        """Estimate pupil/iris center & radii from mask using contours.

        For now, we approximate:
          - center = centroid of mask
          - iris_radius = radius of circle with same area
          - pupil_radius = 0.5 * iris_radius (rough guess)
        For production, you should detect pupil separately.
        """
        ys, xs = np.where(iris_mask > 0)
        if len(xs) == 0:
            h, w = iris_mask.shape[:2]
            return (w // 2, h // 2), min(h, w) // 8, min(h, w) // 4

        cx = float(xs.mean())
        cy = float(ys.mean())
        area = float(len(xs))
        iris_radius = int((area / np.pi) ** 0.5)
        pupil_radius = max(iris_radius // 2, 1)
        return (int(cx), int(cy)), pupil_radius, iris_radius
