import cv2
import numpy as np
import os
from typing import List, Optional, Tuple
from ..base.interfaces import BaseFaceDetector
from ..base.schemas import BoundingBox, DetectedFace, ImageType
from ..base.exceptions import DetectionError


class HaarCascadeDetector(BaseFaceDetector):
    """Traditional face detector using Haar cascades"""

    def __init__(
            self,
            cascade_path: Optional[str] = None,
            min_size: tuple[int, int] = (20, 20),
            scale_factor: float = 1.05,
            min_neighbors: int = 5
    ):
        """
        Initialize Haar cascade detector
        :args
            cascade_path: str: Path to Haar cascade XML file
            min_size: tuple[int, int]: Minimum size of face
            scale_factor: float: Scale factor for detection
            min_neighbors: int: Minimum neighbors for detection
        """
        super().__init__()
        self.name = "haar_cascade"
        self.input_type = ImageType.BGR

        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better detection
        :args
            image: BGR image
        :return
            Preprocessed grayscale image
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Enhance contrast
            gray = cv2.equalizeHist(gray)

            # Reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            return gray
        except Exception as e:
            raise DetectionError(f"Error in preprocessing image: {e}")

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in image
        :args
            image: BGR image
        :returns:
            List of detected faces
        """
        try:
            if image is None or image.size == 0:
                raise DetectionError("Invalid input image")

            # Preprocess image
            gray = self.preprocess_image(image)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size
            )

            # Convert to DetectedFace objects
            detected_faces = []

            for (x, y, w, h) in faces:
                bbox = BoundingBox(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h)
                )

                # Extract face ROI with padding
                face_roi = self._extract_face_roi(image, (x, y, w, h))

                # Create DetectedFace object
                detected_face = DetectedFace(
                    bbox=bbox,
                    image=face_roi
                )

                detected_faces.append(detected_face)
            return detected_faces

        except Exception as e:
            raise DetectionError(f"Error in detecting faces: {e}")

    def detect_single(self, image: np.ndarray) -> DetectedFace:
        return super().detect_single(image)

    @staticmethod
    def _extract_face_roi(
            image: np.ndarray,
            bbox: Tuple[int, int, int, int],
            padding: float = 0.2
    ) -> np.ndarray:
        """
        Extract face region from image with padding
        :args
            image: BGR image
            bbox: Face bounding box (x, y, w, h)
            padding: Padding around face
        :return
            Face region
        """
        x, y, w, h = bbox

        # Calculate padding
        pad_x = int(w * padding)
        pad_y = int(h * padding)

        # Calculate ROI coordinates with padding
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, image.shape[1])
        y2 = min(y + h + pad_y, image.shape[0])

        # Extract and resize ROI
        face_roi = image[y1:y2, x1:x2]

        face_roi = cv2.resize(face_roi, (160, 160))

        return face_roi


if __name__ == "__main__":
    # Initialize detector
    detector = HaarCascadeDetector()

    # Read test image
    img = cv2.imread("./data/test_images/tienduy.jpg")
    if img is None:
        print("Image not found")
        exit(0)

    # Detect faces
    detected_faces_list = detector.detect(img)

    # Draw results
    for detected_face in detected_faces_list:
        bbox = detected_face.bbox
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display image
    cv2.imshow("Detected Faces", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Save extracted faces
    # for i, detected_face in enumerate(detected_faces_list):
    #     if detected_face.image is not None:
    #         face_roi = detected_face.image
    #         cv2.imwrite(f"face_{i}.jpg", face_roi)
