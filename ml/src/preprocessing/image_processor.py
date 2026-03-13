import cv2
import numpy as np

def create_banana_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    강력한 배경 억제 필터와 컨투어 검출을 결합하여 바나나 영역을 정확하게 추출합니다.
    """
    h, w = image_bgr.shape[:2]
    total_area = h * w
    
    # 1. 정밀한 HSV 임계값 적용 (배경의 저채도 영역 배제)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # 노란색: 채도(S)와 명도(V) 하한선을 높여 칙칙한 나무색 배제
    lower_yellow = np.array([10, 60, 60])
    upper_yellow = np.array([45, 255, 255])
    mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    
    # 녹색: 채도 하한선 조정
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([95, 255, 255])
    mask_green = cv2.inRange(image_hsv, lower_green, upper_green)
    
    # 갈색/어두운 점: 명도 범위를 좁히고 채도를 높임
    lower_brown = np.array([0, 40, 30])
    upper_brown = np.array([25, 255, 160])
    mask_brown = cv2.inRange(image_hsv, lower_brown, upper_brown)
    
    mask = cv2.bitwise_or(mask_yellow, mask_green)
    mask = cv2.bitwise_or(mask, mask_brown)
    
    # 2. 모폴로지 연산으로 파편 제거 및 결합
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. 고급 컨투어 필터링
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800: # 너무 작은 노이즈 제거
            continue
            
        # Geometric Filter 1: 이미지 전체를 덮는 거대 객체(배경) 제외
        if area > total_area * 0.8:
            continue
            
        # Geometric Filter 2: 이미지 경계면에 너무 많이 닿아 있는 객체 제외
        x, y, bw, bh = cv2.boundingRect(cnt)
        on_border = (x <= 2 or y <= 2 or x + bw >= w - 3 or y + bh >= h - 3)
        # 바나나가 살짝 닿을 순 있지만, 테두리를 따라 길게 형성된 것은 배경
        if on_border and area > total_area * 0.5:
            continue
            
        # 중앙 점수 계산 (중심에 가까울수록 가산점)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist_to_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
            # 면적은 크고 중앙에 가까운 것을 선호
            score = area - (dist_to_center * 5)
            candidate_contours.append((cnt, score))
    
    final_mask = np.zeros_like(mask)
    if candidate_contours:
        # 점수가 가장 높은 후보 선정
        candidate_contours.sort(key=lambda x: x[1], reverse=True)
        best_cnt = candidate_contours[0][0]
        # 컨투어 내부를 꽉 채움
        cv2.drawContours(final_mask, [best_cnt], -1, 255, thickness=cv2.FILLED)
            
    return final_mask

def extract_features(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    마스킹된 영역에서 특징(Mean, Std of HSV)을 추출합니다.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels = image_hsv[mask > 0]
    
    if len(pixels) == 0:
        return np.zeros(6)
    
    mean_hsv = np.mean(pixels, axis=0)
    std_hsv = np.std(pixels, axis=0)
    features = np.concatenate([mean_hsv, std_hsv])
    
    return features

def process_image(image_bgr: np.ndarray) -> np.ndarray:
    """
    전체 전처리 파이프라인을 실행합니다.
    """
    mask = create_banana_mask(image_bgr)
    features = extract_features(image_bgr, mask)
    return features
