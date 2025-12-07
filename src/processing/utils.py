import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def list_images(folder):
    """Return a list of image paths in the folder."""
    folder = Path(folder)
    return [p for p in folder.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]

def load_image(path, grayscale=True):
    """Load an image and convert to grayscale or RGB using OpenCV."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_pil_image(path):
    """Alternative: Load an image using PIL."""
    return Image.open(path)

def compute_blur_variance(image):
    """Variance of Laplacian → measure blur."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def ensure_dir(path: Path):
    """Create folder if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
