# 팀원 1 (머신러닝 베이스라인) 사용 안내서

프론트엔드(Streamlit 등)에서 바나나 상태 분류기(kNN 모델)를 앱으로 연동하기 위해 필요한 에셋 파일과 파이프라인(전처리, 특징 추출, 추론)에 대한 가이드라인입니다. 

---

## 1. 📦 필수 에셋 파일 (Models)

학습이 완료된 kNN 모델을 추론 시 넘겨서 결과를 받아오려면 아래 두 가지 `pkl` 가중치 파일이 반드시 필요합니다. 해당 파일들은 `models/saved/` 경로 내에 위치해 있습니다.

1.  **`knn_model.pkl`**: 가장 정확도가 높았던 Hyperparameter(K=5 등)로 학습이 완료된 K-Nearest Neighbors 분류 모델 객체입니다. 
2.  **`scaler.pkl`**: 이미지의 추출된 특징 값들을 동일한 숫자 스케일로 맞추기 위한 `StandardScaler` 객체입니다. (이 스케일러를 거치지 않으면 전혀 다른 추론 결과가 나옵니다.)

이 파일들을 앱 구동 디렉토리에 함께 옮긴 후 불러와야 합니다.

---

## 2. ⚙️ 데이터 전처리 파이프라인 (Data Pipeline)

사용자가 새롭게 등록한 1장의 바나나 사진을 kNN 모델이 인식할 수 있는 **516차원 1D 특징 벡터**로 변환해야 합니다. `data/preprocess.py`와 `data/extract_features.py` 의 소스코드를 복사하여 다음과 같은 단계로 처리합니다.

### Step 1: 비율 유지 리사이즈 (OpenCV)
*   **사용 함수:** `data/preprocess.py` 안의 `pad_and_resize(image_path)`
*   **역할:** 업로드된 이미지의 원본 비율(Aspect Ratio)를 해치지 않게 조정한 후, 부족한 여백은 검정색 패딩(`RGB [0, 0, 0]`)으로 채워 모델 입력 고정 사이즈인 **`224x224` 해상도 다차원 배열 `(224, 224, 3)`** 로 만듭니다.

### Step 2: 516개 특징 점포인트 추출
*   **사용 함수:** `data/extract_features.py` 안의 
    *   `extract_color_histogram(image)` : 조명 변화에 덜 민감한 HSV 모델 변환 후 (8x8x8) 형태의 색상 히스토그램 **512개 변수 추출**
    *   `extract_texture_features(image)`: 밝기 통계(Mean, Std)와 노이즈/반점 식별용 필터(Laplacian, Sobel)를 거쳐 **4개 변수 추출**
*   **역할:** 위 과정으로 나온 총 516개의 숫자들을 하나의 가로 벡터 (예: Numpy shape `[1, 516]`) 로 합칩니다.

---

## 3. 🚀 최종 추론 코드 (Predict Logic 예시)

Streamlit 화면에서 사용자가 사진을 올렸을 때, 내부 서버(앱)에서 처리해야 할 코드의 뼈대 예시입니다.

```python
import joblib
import cv2
import numpy as np

# 1. 앱 시작 시 가중치 불러오기 (매번 불러오면 느려지므로 캐싱 또는 최초 1회 로드 권장)
model = joblib.load("models/saved/knn_model.pkl")
scaler = joblib.load("models/saved/scaler.pkl")

# [가정] image_path는 사용자가 올린 이미지 임시 저장 경로
# 2. 이미지 불러오기 및 224x224 리사이즈 (패딩 포함)
preprocessed_image = pad_and_resize(image_path)  # data/preprocess.py 활용

if preprocessed_image is not None:
    # 3. 516차원 특징 벡터 추출 (가로로 합치기)
    color_features = extract_color_histogram(preprocessed_image) # shape: (512,)
    texture_features = extract_texture_features(preprocessed_image) # shape: (4,)
    
    # 두 1D 배열을 이어 하나로 결합. (kNN 입력을 위해 Batch 차원 [1, 516] 형태로 변환)
    feature_vector = np.concatenate([color_features, texture_features]).reshape(1, -1)

    # 4. 스케일링 (필수⭐️: 학습 때 생성한 분포를 새로운 데이터에도 적용)
    scaled_features = scaler.transform(feature_vector)

    # 5. 추론 결과 (Predict)
    predicted_label = model.predict(scaled_features)[0]   # 'ripe', 'unripe' 등
    
    print(f"이 바나나는 현재 {predicted_label} 상태입니다!")
```

위 가이드에 맞춰 제공된 코드와 모델 파일을 사용하면 곧바로 프론트엔드 연동을 진행할 수 있습니다. 추가적인 수정사항이나 버그가 있다면 언제든 알려주세요!
