import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

def pad_and_resize(image_path: str, target_size=(224, 224)):
    """
    이미지를 불러와 원본 비율을 유지하며 패딩(검은여백)을 적용하고 리사이즈하는 전처리 함수.
    
    Args:
        image_path (str): 이미지 파일 경로
        target_size (tuple): 최종 리사이즈 크기 (기본값: 224x224)
        
    Returns:
        np.ndarray: 전처리된 이미지 배열 (RGB, target_size), 실패 시 None
    """
    # 1. 한글 경로를 고려하여 imdecode 사용
    img_array = np.fromfile(image_path, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"Warning: Cannot read image {image_path}")
        return None
        
    # OpenCV는 기본적으로 BGR로 읽어오므로 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w, _ = image.shape
    target_w, target_h = target_size
    
    # 2. 비율 유지 스케일 계산
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 3. 리사이즈 (Interpolation: 축소 시 INTER_AREA 권장)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 4. 빈 공간 패딩 (검은색 [0, 0, 0])
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    
    return padded_image

def process_and_save_dataset(raw_dir: str, processed_dir: str, target_size=(224, 224), seed=42):
    """
    dataset/raw 안의 원본 데이터를 8:1:1로 분할하고,
    비율 유지 패딩 & 리사이즈를 거친 뒤 dataset/processed 폴더에 저장합니다.
    """
    
    # 클래스명 읽어오기 (보통 폴더명 단위)
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory '{raw_dir}' does not exist.")
        return
        
    classes = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if not classes:
        print(f"Error: No class folders found in '{raw_dir}'.")
        return
        
    print(f"클래스 감지 완료: {classes}")
    
    for cls in classes:
        cls_dir = os.path.join(raw_dir, cls)
        # 이미지 파일들만 읽어오기
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if not images:
            print(f"Warning: No images found in '{cls_dir}'")
            continue
            
        print(f"[{cls}] 클래스({len(images)}장) 분할 및 전처리 시작...")
        
        # 1. 8:1:1 (Train:Val:Test) 데이터 분할
        # 전체 10에서 2(Val+Test)를 떼어내어 Train(80%)과 Temp(20%)로 분할
        try:
            train_imgs, temp_imgs = train_test_split(images, test_size=0.2, random_state=seed)
            # Temp(20%)를 절반(10%씩)으로 나누어 Val과 Test 구성
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=seed)
        except ValueError as e:
            # 데이터 개수가 너무 적을 경우 에러 방지
            print(f"[{cls}] 데이터가 너무 적어 분할할 수 없습니다. 전체를 Train으로 사용합니다.")
            train_imgs, val_imgs, test_imgs = images, [], []

        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        # 2. 로드 -> 전처리 -> 각 폴더에 저장
        for split_name, img_names in splits.items():
            if not img_names:
                continue
                
            # 예: dataset/processed/train/ripe
            dest_dir = os.path.join(processed_dir, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            
            for img_name in tqdm(img_names, desc=f"{cls} - {split_name}"):
                src_path = os.path.join(cls_dir, img_name)
                dest_path = os.path.join(dest_dir, img_name)
                
                # 확장자 변경: 원활한 파이토치 DataLoader를 위해 .jpg로 통일
                name_without_ext = os.path.splitext(img_name)[0]
                dest_path = os.path.join(dest_dir, f"{name_without_ext}.jpg")

                # 전처리 수행
                processed_img = pad_and_resize(src_path, target_size)
                
                if processed_img is not None:
                    # 저장할 때는 다시 RGB -> BGR 변환 (OpenCV 기준)
                    processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                    
                    # 한글 인코딩 문제를 피하기 위해 imencode 사용
                    result, encode_img = cv2.imencode('.jpg', processed_img_bgr)
                    if result:
                        with open(dest_path, mode='w+b') as f:
                            encode_img.tofile(f)
                            
    print("\n--- 전체 전처리 및 저장 완료 ---")
