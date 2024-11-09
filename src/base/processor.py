from typing import Dict, Optional, List
import numpy as np
from .interfaces import BaseFaceDetector, BaseFaceEmbedder, BaseFaceMatcher
from .schemas import DetectedFace, FaceEmbedding, MatchResult


class FaceProcessor:
    """Main class that combines detector, embedder and matcher"""

    def __init__(self,
                 detector: BaseFaceDetector,
                 embedder: BaseFaceEmbedder,
                 matcher: BaseFaceMatcher):
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.database: Dict[str, FaceEmbedding] = {}

    def register_face(self, face_id: str, face_image: np.ndarray) -> bool:
        """Register face in database"""

        # Detect a single face in the image
        detected_face = self.detector.detect_single(face_image)
        if detected_face is None:
            return False

        # Compute embedding
        if detected_face.image is not None:
            embedding = self.embedder.compute_embedding(detected_face.image)
            self.database[face_id] = embedding
            return True
        return False

    def recognize_face(self, face_image: np.ndarray) -> Optional[MatchResult]:
        """Recognize a face against database"""

        # Detect a single face in the image
        detected_face = self.detector.detect_single(face_image)
        if detected_face is None:
            return None

        # Compute embedding and match if available
        if detected_face.image is not None:
            embedding = self.embedder.compute_embedding(detected_face.image)
            match_result = self.matcher.match(embedding, self.database)
            return match_result
        return None
