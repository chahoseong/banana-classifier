import cv2
import numpy as np
import os
import sys

# 현재 디렉토리를 경로에 추가하여 image_processor를 import 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import image_processor

def test_on_sample():
    # 샘플 이미지 경로 (Unripe 폴더에서 하나 선택)
    sample_dir = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\dataset\raw\Unripe"
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.jpg')]
    
    if not sample_files:
        print("샘플 이미지를 찾을 수 없습니다.")
        return

    sample_path = os.path.join(sample_dir, sample_files[0])
    print(f"테스트 이미지: {sample_path}")

    # 이미지 로드
    img = cv2.imread(sample_path)
    if img is None:
        print("이미지를 로드할 수 없습니다.")
        return

    # 300x300 리사이즈 (BE 규격)
    img_resized = cv2.resize(img, (300, 300))

    # 전처리 실행
    features = image_processor.process_image(img_resized)

    print("\n[추출된 특징 벡터]")
    print(f"평균 (H, S, V): {features[:3]}")
    print(f"표준편차 (H, S, V): {features[3:]}")
    
    # 결과 시각화 준비 (마스크 확인용)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    mask = image_processor.create_banana_mask(hsv)
    masked_img = cv2.bitwise_and(img_resized, img_resized, mask=mask)

    # 마스킹된 픽셀 수 확인
    pixel_count = np.sum(mask > 0)
    print(f"\n마스킹된 영역 픽셀 수: {pixel_count} / {300*300}")

    if pixel_count > 0:
        print("성공: 바나나 영역이 감지되었습니다.")
    else:
        print("경고: 바나나 영역이 감지되지 않았습니다. HSV 임계값 조정이 필요할 수 있습니다.")

if __name__ == "__main__":
    test_on_sample()
