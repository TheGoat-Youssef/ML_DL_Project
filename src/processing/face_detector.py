import cv2
import numpy as np
from mtcnn import MTCNN
from pathlib import Path
from .utils import ensure_dir

class FaceDetector:
    def __init__(self, detector="mtcnn"):
        self.detector_name = detector.lower()

        if self.detector_name == "haar":
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(haar_path)

        elif self.detector_name == "mtcnn":
            self.detector = MTCNN()

        else:
            raise ValueError("Detector must be 'haar' or 'mtcnn'")

    def detect(self, image):
        """Retourne (x, y, w, h) ou None."""
        if self.detector_name == "haar":
            faces = self.detector.detectMultiScale(image, 1.3, 5)
            return faces[0] if len(faces) > 0 else None

        elif self.detector_name == "mtcnn":
            faces = self.detector.detect_faces(image)
            if len(faces) == 0:
                return None
            x, y, w, h = faces[0]['box']
            return (x, y, w, h)

    def crop_face(self, image):
        face = self.detect(image)
        if face is None:
            return None

        x, y, w, h = face
        return image[y:y+h, x:x+w]
