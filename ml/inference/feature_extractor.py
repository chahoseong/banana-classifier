import cv2
import numpy as np


DEFAULT_IMAGE_SIZE = (300, 300)


def pad_and_resize(image: np.ndarray, target_size=DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """Resize an image with padding while preserving the aspect ratio."""
    h, w, _ = image.shape
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    return cv2.copyMakeBorder(
        resized_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def extract_color_histogram(image: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
    """Extract a normalized HSV color histogram."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """Extract simple grayscale texture statistics."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_val, std_val = cv2.meanStdDev(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mean = np.mean(sobel_mag)

    return np.array([mean_val[0][0], std_val[0][0], laplacian_var, sobel_mean])


def get_knn_features(image: np.ndarray) -> np.ndarray:
    """Return the combined feature vector used by the baseline kNN model."""
    preprocessed = pad_and_resize(image)
    color_hist = extract_color_histogram(preprocessed)
    texture_feat = extract_texture_features(preprocessed)
    return np.concatenate([color_hist, texture_feat]).reshape(1, -1)
