import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add current folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from .utils import load_image, compute_blur_variance, ensure_dir, list_images
from .face_detector import FaceDetector
from .augmentations import apply_random_augment
import config

class Preprocessor:
    def __init__(self):
        self.detector = FaceDetector(config.FACE_DETECTOR)

    def process_image(self, img_path):
        img = load_image(img_path, grayscale=False)
        if img is None:
            return None, "corrupt"

        # Ensure 3 channels
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Size check
        if img.shape[0] < config.MIN_FACE_SIZE or img.shape[1] < config.MIN_FACE_SIZE:
            return None, "too_small"

        # Detect face
        face = self.detector.crop_face(img)
        if face is None or face.size == 0:
            return None, "no_face"

        # Blur check
        blur = compute_blur_variance(face)
        if blur < config.BLUR_THRESHOLD:
            return None, "blurry"

        # Resize to final model input
        face = cv2.resize(face, config.IMG_SIZE)
        return face, "ok"

    def run(self):
        X, y, errors = [], [], []

        for class_name in config.CLASS_NAMES:
            class_dir = config.RAW_DIR / "train" / class_name

            for img_path in tqdm(list_images(class_dir), desc=f"{class_name}"):
                img, status = self.process_image(img_path)

                if status != "ok":
                    errors.append({"image": str(img_path), "error": status})
                    continue

                # Add clean image
                X.append(img)
                y.append(class_name)

                # Augmentations
                if config.AUGMENTATION_ENABLED:
                    for _ in range(config.AUG_PER_IMAGE):
                        X.append(apply_random_augment(img))
                        y.append(class_name)

        # Convert
        X = np.array(X)
        y = np.array(y)

        # Safety mask (removes None)
        mask = np.array([img is not None for img in X])
        X = X[mask]
        y = y[mask]

        # Ensure output directory exists
        ensure_dir(config.PROCESSED_DIR)

        # Save cleaned dataset
        np.savez(config.PROCESSED_NPZ, X=X, y=y, classes=config.CLASS_NAMES)

        # SAVE PROCESSING ERRORS
        errors_path = config.PROCESSED_DIR / "processing_errors.json"
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=4)

        print(f"\nSaved: {errors_path}")

        return X, y, errors
