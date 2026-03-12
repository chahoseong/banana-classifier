import cv2
import numpy as np

def preprocess_image(image_path: str, target_size=(224, 224)):
    """
    이미지를 불러와 비율을 유지하며 패딩을 적용하고 리사이즈하는 전처리 함수.
    [팀원 1 담당 영역]
    
    Args:
        image_path (str): 이미지 파일 경로
        target_size (tuple): 최종 리사이즈 크기 (기본값: 224x224)
        
    Returns:
        np.ndarray: 전처리된 이미지 배열
    """
    # TODO: OpenCV를 사용한 패딩 및 리사이즈 로직 구현
    pass
