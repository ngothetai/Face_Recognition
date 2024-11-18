import cv2
import numpy as np
from typing import List, Tuple

from skimage.exposure import histogram
from skimage.feature import local_binary_pattern, hog
from ..base.interfaces import BaseFaceEmbedder
from ..base.schemas import FaceEmbedding, ImageType
from ..base.exceptions import EmbeddingError


class LBPEmbedder(BaseFaceEmbedder):
    """
    Face embedder using Local Binary Patterns (LBP)
    """
    def __init__(
        self,
        radius: int = 3,
        n_points: int = 8,
        grid_size: Tuple[int, int] = (8, 8),
        normalize: bool = True,
    ):
        """
        Initialize LBP embedder
        :args
            radius: int: Radius for LBP
            n_points: int: Number of points for LBP
            grid_size: Tuple[int, int]: Grid size for LBP
            normalize: bool: Normalize LBP histogram
        """
        super().__init__()
        self.name = "lbp_embedder"
        self.input_type = ImageType.BGR

        self.radius = radius
        self.n_points = n_points
        self.grid_size = grid_size
        self.normalize = normalize

        n_bina = n_points + 2
        self.embedding_size = n_bina * grid_size[0] * grid_size[1]

    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for embedding
        :args
            face_image: BGR face image
        :return
            Preprocessed face image
        """
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Normalize Lighting
            face_image = cv2.equalizeHist(face_image)

            return face_image
        except Exception as e:
            raise EmbeddingError(f"Error preprocessing face image: {str(e)}")

    def compute_embedding(self, face_image: np.ndarray) -> FaceEmbedding:
        """
        Compute LBP embedding for face image
        :args
            face_image: BGR face image
        :return
            FaceEmbedding object
        """

        try:
            # Preprocess image
            face_image = self.preprocess_face(face_image)

            # Compute LBP
            lbp = local_binary_pattern(
                image=face_image,
                P=self.n_points,
                R=self.radius,
                method='uniform'
            )

            # Split image into grid
            cell_height = face_image.shape[0] // self.grid_size[0]
            cell_width = face_image.shape[1] // self.grid_size[1]

            # Compute LBP histogram for each cell
            histograms = []
            n_bins = self.n_points + 2

            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    # Get cell coordinates
                    x_start = i * cell_height
                    x_end = (i + 1) * cell_height
                    y_start = j * cell_width
                    y_end = (j + 1) * cell_width

                    # Extract cell
                    cell = lbp[y_start:y_end, x_start:x_end]

                    # Compute histogram
                    hist, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins))

                    histograms.extend(hist)

            # Convert to numpy array
            embedding = np.array(histograms, dtype=np.float32)

            # Normalize if required
            if self.normalize:
                embedding /= np.linalg.norm(embedding)

            return FaceEmbedding(
                vector=embedding,
                confidence=1.0 # LBP does not have confidence
            )
        except Exception as e:
            raise EmbeddingError(f"Error computing LBP embedding: {str(e)}")


class HOGEmbedder(BaseFaceEmbedder):
    """Face embedder using Histogram of Oriented Gradients"""

    def __init__(
            self,
            orientations: int = 9,
            pixels_per_cell: Tuple[int, int] = (8, 8),
            cells_per_block: Tuple[int, int] = (2, 2),
            normalize: bool = True
    ):
        """
        Initialize HOG embedder
        args:
            orientations: Number of orientation bins
            pixels_per_cell: Size of cell for gradient computation
            cells_per_block: Number of cells in each block
            normalize: Whether to normalize features
        """
        super().__init__()
        self.name = "hog_embedder"
        self.input_type = ImageType.BGR

        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.normalize = normalize
        self.embedding_size = None

    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for HOG
        Args:
            face_image: BGR face image
        Returns:
            Preprocessed grayscale image
        """
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image

            # Normalize contrast
            gray = cv2.equalizeHist(gray)

            return gray

        except Exception as e:
            raise EmbeddingError(f"Error in face preprocessing: {str(e)}")

    def compute_embedding(self, face_image: np.ndarray) -> FaceEmbedding:
        """
        Compute HOG embedding for face image
        Args:
            face_image: BGR face image
        Returns:
            FaceEmbedding object
        """
        try:
            # Preprocess image
            gray = self.preprocess_face(face_image)

            # Compute HOG features
            hog_features = hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                visualize=False,
                feature_vector=True,
                block_norm='L2-Hys'
            )

            # Convert to float32
            embedding_vector = np.array(hog_features, dtype=np.float32)
            self.embedding_size = embedding_vector.shape[0]

            # Normalize if requested
            if self.normalize:
                embedding_vector = embedding_vector / (np.linalg.norm(embedding_vector) + 1e-7)

            return FaceEmbedding(
                vector=embedding_vector,
                confidence=1.0  # HOG doesn't provide confidence score
            )

        except Exception as e:
            raise EmbeddingError(f"HOG embedding computation failed: {str(e)}")