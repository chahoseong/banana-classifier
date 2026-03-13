
# 🍌 Banana Ripe Classifier - TRD (Technical Requirements Document)

이 문서는 바나나 숙성도 분류 시스템의 기술적 설계와 구현 세부 사항을 정의합니다.

## 1. 시스템 아키텍처

시스템은 모듈화된 설계를 통해 프론트엔드와 추론 엔진을 분리하여 유지보수와 확장을 용이하게 합니다.

- **Frontend:** Streamlit 기반 웹 인터페이스
- **Inference Engine:** 모델 로딩 및 이미지 전처리를 담당하는 코어 모듈
- **Models:** ML 베이스라인 모델 및 전이학습 기반 딥러닝 모델 (ResNet/MobileNet)

## 2. 기술 스택

| 구분 | 기술 / 라이브러리 | 버전 / 상세 |
|---|---|---|
| **언어** | Python | **3.13.11** |
| **프론트엔드** | Streamlit | 최신 안정 버전 |
| **이미지 처리** | OpenCV (opencv-python) | 이미지 캡처 및 전처리 |
| | Pillow (PIL) | 이미지 형식 변환 및 출력 |
| **ML/DL 프레임워크** | PyTorch | 딥러닝 모델 학습 및 추론 |
| | scikit-learn | ML 베이스라인 모델 및 평가 |
| | LightGBM | 정형 데이터 기반 분류 (선택 사항) |
| **데이터 처리** | NumPy | 수치 계산 및 배열 처리 |
| | Pandas | 데이터 매니페스트 및 결과 분석 |

## 3. 데이터 흐름

1. **Input:** 사용자가 Streamlit 웹캠 인터페이스에 바나나를 노출. `st.camera_input`이 이미지를 캡처.
2. **Preprocessing:** 캡처된 이미지를 `InferenceEngine`으로 전달. 모델 요구사항에 맞춰 리사이징 및 정규화 수행.
3. **Inference:** 선택된 모델(BaseModel 상속)이 이미지에 대한 숙성도 카테고리 및 신뢰도 예측.
4. **Output:** 예측 결과(unripe, ripe, overripe, dispose)와 신뢰도를 사용자 UI에 시각적으로 표시.

## 4. 설치 및 설정

### 환경 구축
```powershell
# 가상환경 생성 (Python 3.13.11 기준)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
```

### 실행
```powershell
streamlit run src/app/main.py
```
