import cv2
import numpy as np

def pad_and_resize(image: np.ndarray, target_size=(224, 224)):
    """
    이미지의 원본 비율을 유지하며 패딩을 적용하고 리사이즈합니다.
    (data/preprocess.py 로직 통합)
    """
    h, w, _ = image.shape
    target_w, target_h = target_size
    
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    return padded_image

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    HSV 색상 공간에서 3D 컬러 히스토그램 추출 (512차원)
    (data/extract_features.py 로직 통합)
    """
    # RGB로 입력받으므로 BGR로 변환 후 HSV로 처리 (OpenCV 호환성)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    
    return hist.flatten()

def extract_texture_features(image):
    """
    픽셀 통계 및 엣지 강도 특징 추출 (4차원)
    (data/extract_features.py 로직 통합)
    """
    # RGB -> GRAY
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    mean_val, std_val = cv2.meanStdDev(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mean = np.mean(sobel_mag)
    
    return np.array([mean_val[0][0], std_val[0][0], laplacian_var, sobel_mean])

def get_knn_features(image: np.ndarray) -> np.ndarray:
    """
    kNN 모델 입력을 위한 통합 516차원 특징 벡터를 반환합니다.
    """
    # 1. 전처리 (224x224)
    preprocessed = pad_and_resize(image)
    
    # 2. 특징 추출
    color_hist = extract_color_histogram(preprocessed)
    texture_feat = extract_texture_features(preprocessed)
    
    # 3. 결합 (1, 516)
    features = np.concatenate([color_hist, texture_feat]).reshape(1, -1)
    return features
