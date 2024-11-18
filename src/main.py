import cv2
from .traditional.face_detector import HaarCascadeDetector
from .traditional.face_embedder import LBPEmbedder, HOGEmbedder
from .base.faiss_vector_store import FAISSEmbeddingVectorStore
from .base.processor import FaceProcessor
from .base.schemas import MatchResult

def main():
    # Initialize components
    detector = HaarCascadeDetector()
    embedder = LBPEmbedder()  # or HOGEmbedder()
    vector_store = FAISSEmbeddingVectorStore(embedding_dim=embedder.embedding_size)
    processor = FaceProcessor(detector, embedder, vector_store)

    # Register a face
    face_id = "person_1"
    face_image = cv2.imread("./data/test_images/tienduy.jpg")
    if face_image is None:
        print("Image not found")
        return

    if processor.register_face(face_id, face_image):
        print(f"Face {face_id} registered successfully.")
    else:
        print(f"Failed to register face {face_id}.")

    # Recognize a face
    query_image = cv2.imread("./data/test_images/cotienduy.jpg")
    if query_image is None:
        print("Query image not found")
        return

    match_result = processor.recognize_face(query_image)
    if match_result and match_result.matched:
        print(f"Face recognized with ID: {match_result.matched_id}, Confidence: {match_result.confidence}")
    else:
        print("Face not recognized.")

if __name__ == "__main__":
    main()
