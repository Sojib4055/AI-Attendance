from .iris_detector import IrisDetector
from .iris_segmenter import IrisSegmenter
from .normalization import DaugmanNormalizer
from .encoder import IrisEncoder

class IrisPipeline:
    def __init__(self, cfg):
        self.detector = IrisDetector(cfg["models"]["iritrack_cascade"])
        self.segmenter = IrisSegmenter()
        self.normalizer = DaugmanNormalizer(
            radial_res=cfg["norm"]["radial_res"],
            angular_res=cfg["norm"]["angular_res"],
        )
        self.encoder = IrisEncoder(
            cfg["models"].get("deepirisnet2", None),
            device=cfg.get("device", "cuda"),
        )

    def process_eye(self, eye_crop):
        mask = self.segmenter.segment(eye_crop)
        pupil_center, pupil_radius, iris_radius = self.normalizer.estimate_geometry_from_mask(mask)
        norm_iris = self.normalizer.normalize(
            eye_crop, mask, pupil_center, pupil_radius, iris_radius
        )
        emb = self.encoder.encode(norm_iris)
        return emb

    def process_frame(self, frame_bgr):
        eyes = self.detector.detect_eyes(frame_bgr)
        embeddings = []
        for eye in eyes:
            try:
                emb = self.process_eye(eye["eye_crop"])
                embeddings.append(
                    {
                        "embedding": emb,
                        "bbox": eye["bbox"],
                        "confidence": eye["confidence"],
                    }
                )
            except Exception:
                continue
        return embeddings
