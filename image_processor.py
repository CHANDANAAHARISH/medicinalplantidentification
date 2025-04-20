import cv2
import numpy as np

def preprocess_image(image):
    """Enhanced image preprocessing optimized for plant features"""
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3,3), 0)

    # Resize to match training size (increased for better detail)
    resized = cv2.resize(blurred, (128, 128))

    # Normalize pixel values
    normalized = resized.astype(np.float32) / 255.0

    return normalized