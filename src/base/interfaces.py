from abc import ABC, abstractclassmethod, abstractmethod
from typing import List, Optional, Dict
import numpy as np
from .schemas import (
    BoundingBox,
    FaceEmbedding,
    DetectedFace,
    MatchResult,
    ImageType
)


class BaseFaceDetector(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        self.name: str = "base_detector"
        self.input_type: ImageType = ImageType.BGR

    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before detection
        :arg
            image: np.ndarray: Input image
        :return
            np.ndarray: Preprocessed image
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in image
        :arg
            image: np.ndarray: Input image
        :return
            List[DetectedFace]: List of detected faces
        """
        pass

    @abstractmethod
    def detect_single(self, image: np.ndarray) -> DetectedFace:
        """
        Detect single face in image
        :arg
            image: np.ndarray: Input image
        :return
            DetectedFace: Detected face
        """
        faces = self.detect(image)
        return faces[0] if len(faces) == 1 else None


class BaseFaceEmbedder(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize face embedder with configuration"""
        self.name: str = "base_embdder"
        self.input_type: ImageType = ImageType.BGR
        self.embedding_size: int = 0

    @abstractmethod
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image before embedding
        """
        pass

    @abstractmethod
    def compute_embedding(self, face_image: np.ndarray) -> FaceEmbedding:
        """
        Compute embedding for a face image
        :args
            face_image: np.ndarray: Face image
        :returns
            FaceEmbedding: Face embedding
        """
        pass


class BaseFaceMatcher(ABC):
    """Base interface for face matcher"""

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize face matcher with configuration"""
        self.name: str = "base_matcher"
        self.threshold: float = 0.5

    @abstractmethod
    def compute_similarity(self,
                           embedding1: np.ndarray,
                           embedding2: np.ndarray) -> float:
        """
        Compute similarity score between two face embeddings
        :args
            embedding1: np.ndarray: Face embedding 1
            embedding2: np.ndarray: Face embedding 2
        :returns
            float: Similarity score
        """
        pass

    @abstractmethod
    def match(self,
              query_embedding: FaceEmbedding,
              database_embeddings: Dict[str, FaceEmbedding]) -> MatchResult:
        """
        Match query embedding against database
        :args
            query_embedding: FaceEmbedding: Query face embedding
            database_embeddings: Dict[str, FaceEmbedding]: Database of face embeddings
        :return:
            MatchResult: Match result
        """
        pass