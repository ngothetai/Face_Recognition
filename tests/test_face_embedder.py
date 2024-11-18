import unittest
import numpy as np
import cv2
from src.traditional.face_embedder import LBPEmbedder, HOGEmbedder
from src.base.schemas import FaceEmbedding


class TestLBPEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = LBPEmbedder()
        self.face_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_preprocess_face(self):
        preprocessed_image = self.embedder.preprocess_face(self.face_image)
        self.assertEqual(preprocessed_image.shape, (64, 64))
        self.assertEqual(preprocessed_image.dtype, np.uint8)

    def test_compute_embedding(self):
        embedding = self.embedder.compute_embedding(self.face_image)
        self.assertIsInstance(embedding, FaceEmbedding)
        self.assertEqual(len(embedding.vector), self.embedder.embedding_size)
        self.assertEqual(embedding.confidence, 1.0)


class TestHOGEmbedder(unittest.TestCase):
    def setUp(self):
        self.embedder = HOGEmbedder()
        self.face_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_preprocess_face(self):
        preprocessed_image = self.embedder.preprocess_face(self.face_image)
        self.assertEqual(preprocessed_image.shape, (64, 64))
        self.assertEqual(preprocessed_image.dtype, np.uint8)

    def test_compute_embedding(self):
        embedding = self.embedder.compute_embedding(self.face_image)
        self.assertIsInstance(embedding, FaceEmbedding)
        self.assertEqual(len(embedding.vector), self.embedder.embedding_size)
        self.assertEqual(embedding.confidence, 1.0)

if __name__ == '__main__':
    unittest.main()
