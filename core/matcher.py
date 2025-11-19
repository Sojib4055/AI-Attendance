import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

class IrisMatcher:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def match(self, query_emb, enrolled_templates):
        best_score = -1.0
        best_person = None
        for t in enrolled_templates:
            score = cosine_similarity(query_emb, t["embedding"])
            if score > best_score:
                best_score = score
                best_person = t["person_id"]
        if best_score >= self.threshold:
            return best_person, best_score
        return None, best_score
