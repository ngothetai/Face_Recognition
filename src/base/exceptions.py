class FaceProcessingError(Exception):
    """Base exception for all of face processing errors"""
    pass


class DetectionError(FaceProcessingError):
    """Raised when face detection fails"""
    pass


class EmbeddingError(FaceProcessingError):
    """Raised when embedding computation fails"""
    pass


class MatchingError(FaceProcessingError):
    """Raised when face matching fails"""
    pass


class VectorStoreError(FaceProcessingError):
    """Raised when vector store operation fails"""
    pass
