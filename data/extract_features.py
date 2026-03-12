import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    OpenCV를 이용해 HSV 색상 공간에서 3D 컬러 히스토그램을 추출합니다.
    바나나의 경우 노란색, 초록색, 검은색(반점, 갈변) 등의 색상 정보가 매우 중요합니다.
    """
    # HSV 변환 (HSV가 RGB보다 조명 변화에 강건하고 색상 구분에 유리)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 히스토그램 계산 (H: 0~180, S: 0~256, V: 0~256 범위를 bin 개수로 나눔)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    
    # 정규화 (이미지 크기에 상관없이 동일한 스케일을 가지도록 함)
    cv2.normalize(hist, hist)
    
    # 1D 배열로 평탄화 (특징 벡터)
    return hist.flatten()

def extract_texture_features(image):
    """
    OpenCV와 Numpy를 활용하여 단순 픽셀 통계 및 윤곽선(엣지) 특징을 추출합니다.
    숙성될수록 생기는 반점이나 표면의 질감 변화를 포착합니다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 기본 그레이스케일 통계 값 (밝기 평균, 표준편차)
    mean_val, std_val = cv2.meanStdDev(gray)
    
    # 2. Laplacian 분산 (질감/선명도 변화 척도 - 표면이 거칠고 반점이 많을수록 분산 증가)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 3. Sobel Edge Magnitude 평균 (윤곽선의 뚜렷한 정도)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mean = np.mean(sobel_mag)
    
    return np.array([mean_val[0][0], std_val[0][0], laplacian_var, sobel_mean])

def extract_features_from_dataset(processed_dir, output_csv):
    """
    dataset/processed 폴더 내의 각 세트(train, val, test)와 숙성도별 이미지를 
    탐색하여 특징을 추출하고 라벨과 함께 CSV로 저장합니다.
    """
    data = []
    
    if not os.path.exists(processed_dir):
        print(f"Error: Directory '{processed_dir}' not found. 먼저 전처리 과정을 수행해주세요.")
        return
        
    # train, val, test 폴더 순회
    splits = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    
    for split in splits:
        split_dir = os.path.join(processed_dir, split)
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for label in classes:
            class_dir = os.path.join(split_dir, label)
            # 숨김 파일 제외하고 이미지 파일만 로드
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in tqdm(images, desc=f"Extracting {split}/{label}"):
                img_path = os.path.join(class_dir, img_name)
                
                # 한글 계정명/경로 오류 방지를 위한 imdecode 사용
                img_array = np.fromfile(img_path, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue
                    
                # 각 특징들을 추출
                color_hist = extract_color_histogram(image, bins=(8, 8, 8))
                texture_feat = extract_texture_features(image)
                
                # CSV에 담을 딕셔너리(Row) 생성
                row = {
                    'filename': img_name,
                    'split': split,   # 나중에 머신러닝 학습 시, 데이터 분할을 그대로 쓰기 위함
                    'label': label    # 정답 클래스 (unripe, ripe, overripe, dispose)
                }
                
                # 컬러 히스토그램 (8x8x8 = 512 차원) 컬럼 추가
                for i, val in enumerate(color_hist):
                    row[f'color_hist_{i}'] = val
                
                # 텍스처 (픽셀 통계) 4차원 컬럼 추가
                row['texture_gray_mean'] = texture_feat[0]
                row['texture_gray_std'] = texture_feat[1]
                row['texture_laplacian_var'] = texture_feat[2]
                row['texture_sobel_mean'] = texture_feat[3]
                
                data.append(row)
                
    if data:
        # 데이터프레임으로 변환 후 CSV 저장 (한글 깨짐 방지 utf-8-sig)
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ 특징 추출 완료!")
        print(f"  - 총 데이터 수: {len(df)}개")
        print(f"  - 저장 위치: {output_csv}")
        print(f"  - 특징 구성: Color Histogram ({len(color_hist)}개) + Texture ({len(texture_feat)}개)")
    else:
        print("추출 대상 이미지 데이터가 발견되지 않았습니다.")

if __name__ == "__main__":
    # 프로젝트 최상단 기준
    PROCESSED_DATA_DIR = os.path.join('dataset', 'processed')
    OUTPUT_FILE = 'features.csv'
    
    print("바나나 이미지 특징(Feature) 추출을 시작합니다...")
    extract_features_from_dataset(PROCESSED_DATA_DIR, OUTPUT_FILE)
