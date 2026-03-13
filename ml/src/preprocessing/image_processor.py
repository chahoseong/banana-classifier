import cv2
import numpy as np

def create_banana_mask(image_hsv: np.ndarray) -> np.ndarray:
    """
    HSV 색상 임계값을 사용하여 바나나 영역의 마스크를 생성합니다.
    """
    # 일반적인 노란색/녹색 바나나 영역 HSV 범위
    # 노란색: H(20-30), S(100-255), V(100-255)
    # 녹색: H(35-85), S(50-255), V(50-255)
    
    # 노란색 범위
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    
    # 녹색 범위
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # 갈색/어두운 점 (숙성된 바나나)
    lower_brown = np.array([0, 20, 20])
    upper_brown = np.array([20, 255, 150])

    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(image_hsv, lower_brown, upper_brown)
    
    # 모든 마스크 합치기
    mask = cv2.bitwise_or(mask_yellow, mask_green)
    mask = cv2.bitwise_or(mask, mask_brown)
    
    # 노이즈 제거 (Opening/Closing)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def extract_features(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    마스킹된 영역에서 특징(Mean, Std of HSV)을 추출합니다.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # 마스크가 적용된 픽셀만 추출
    pixels = image_hsv[mask > 0]
    
    if len(pixels) == 0:
        return np.zeros(6) # 특징 추출 실패 시 0 반환
    
    # H, S, V 채널별 평균 및 표준편차 계산
    mean_hsv = np.mean(pixels, axis=0)
    std_hsv = np.std(pixels, axis=0)
    
    # 특징 벡터 결합 [mean_h, mean_s, mean_v, std_h, std_s, std_v]
    features = np.concatenate([mean_hsv, std_hsv])
    
    return features

def process_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    전체 전처리 파이프라인을 실행합니다.
    입력: 300x300 BGR Numpy Array
    출력: 특징 벡터 (Numpy Array)
    """
    # 1. HSV 변환
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. 마스크 생성
    mask = create_banana_mask(image_hsv)
    
    # 3. 특징 추출
    features = extract_features(image_bgr, mask)
    
    return features
