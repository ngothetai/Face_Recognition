from optparse import Option

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Union
import numpy as np
from enum import Enum
import numpy.typing as npt


class ImageType(Enum):
    BGR = "BGR"
    RGB = "RGB"
    GRAY = "GRAY"


class BoundingBox(BaseModel):
    x: int = Field(..., description="Top left x coordinate")
    y: int = Field(..., description="Top left y coordinate")
    width: int = Field(..., description="Width of bounding box")
    height: int = Field(..., description="Height of bounding box")
    confidence: float = Field(default=1.0, description="Detection confidence score")


class FaceEmbedding(BaseModel):
    vector: npt.NDArray[np.float32] = Field(..., description="Face embedding vector")
    confidence: float = Field(default=1.0, description="Confidence score of face embedding")

    class Config:
        arbitrary_types_allowed = True

class DetectedFace(BaseModel):
    bbox: BoundingBox
    embedding: Optional[FaceEmbedding] = None
    landmarks: Optional[dict] = None
    image: Optional[npt.NDArray] = None

    class Config:
        arbitrary_types_allowed = True


class MatchResult(BaseModel):
    matched: bool
    confidence: float
    matched_id: Optional[str] = None
    distance: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

class EmbeddingVectorStore(BaseModel):
    vectors: List[FaceEmbedding] = Field(..., description="List of face embeddings")
    ids: List[str] = Field(..., description="List of corresponding face IDs")

    class Config:
        arbitrary_types_allowed = True
