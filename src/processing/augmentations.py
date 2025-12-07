import cv2
import numpy as np
import random

def random_rotation(image):
    angle = random.uniform(-10, 10)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def random_flip(image):
    return cv2.flip(image, 1)

def random_brightness(image):
    factor = random.uniform(0.7, 1.3)
    return np.clip(image * factor, 0, 255).astype(np.uint8)

def random_noise(image):
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def apply_random_augment(image):
    ops = [random_rotation, random_flip, random_brightness, random_noise]
    op = random.choice(ops)
    return op(image)
