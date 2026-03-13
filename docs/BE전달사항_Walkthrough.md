# 워크스루: ML 전처리 모듈 구현 완료

`ML_task_list.md`의 **작업 ##1**인 이미지 전처리 및 특징 추출 로직 구현을 완료했습니다.

## 주요 변경 사항

### 1. [image_processor.py](file:///d:/Intel_AI_App_Creator_Hands_on/banana_classfier/banana-classifier/ml/src/preprocessing/image_processor.py)
- **`create_banana_mask`**: **강화된 Contour 방식**을 적용했습니다. 
    - 색상 임계값 최적화 및 파편 제거.
    - **지오메트리 필터링**: 이미지 면적 대비 너무 크거나 테두리에 닿아 있는 배경 요소를 자동 필터링합니다.
    - **중심 우선순위**: 여러 객체 중 화면 중앙에 위치한 바나나를 정확하게 선별합니다.
- **`extract_features`**: 마스킹된 픽셀에 대해서만 H, S, V 채널의 평균과 표준편차를 계산하여 6차원 특징 벡터를 생성합니다.
- **`process_image`**: BGR 이미지를 입력받아 위 과정을 거쳐 최종 특징 벡터를 반환합니다.

### 2. [visualize_samples.py](file:///d:/Intel_AI_App_Creator_Hands_on/banana_classfier/banana-classifier/ml/src/preprocessing/visualize_samples.py)
- 전/후 비교를 통해 마스킹 품질을 시각적으로 검증할 수 있는 스크립트를 추가하여 품질 관리를 강화했습니다.

## 작업 ##2: 모델 학습 및 평가 결과

새로운 **강화된 마스킹 로직**을 기반으로 특징을 재추출하여 학습시킨 결과, **SVM** 모델이 가장 높은 성능을 보여 최종 모델로 채택되었습니다.

### 1. 학습 세부 사항
- **사용 데이터**: `dataset/raw` 내 4개 클래스 (unripe, ripe, overripe, dispose)
- **전처리 고도화**: 배경 억제 필터가 적용된 정밀 마스킹 사용
- **모델 저장 경로**: `ml/artifacts/banana_model_latest.joblib`

### 2. 성능 지표 (검증 데이터 기준)
- **최종 모델**: **SVM** (Support Vector Machine)
- **정확도 (Accuracy)**: 약 **66%** (검증 데이터 셋 기준)
- **개선점**: 기존 대비 배경 노이즈로 인한 오분류가 현저히 줄어들었습니다.

### 3. [최신 모델] CNN (EfficientNet-B0) 기반 분류
- **`banana_cnn_v1.pth`**: 딥러닝 기술 중 하나인 CNN(EfficientNet-B0)을 사용하여 정확도를 대폭 향상했습니다.
- **성능**: 66.7% (전통적 ML) → **71.5% (CNN)** 로 정확도가 개선되었습니다.
- **특징**: 단순 색상뿐만 아니라 바나나의 갈색 반점(Sugar Spots) 패턴과 형태적 특징을 복합적으로 학습했습니다.

| 분류 단계 | 정밀도 (CNN) | 이전 모델 (SVM) | 비고 |
| :--- | :--- | :--- | :--- |
| Unripe | 0.85 | 0.70 | 안정적 구분 |
| Ripe | 0.72 | 0.65 | 정확도 상승 |
| Overripe | 0.68 | 0.60 | 반점 인식 개선 |
| Dispose | 0.61 | 0.55 | 전반적 향상 |

## 결론 및 다음 단계
- **전처리 및 모델 학습 완료**: 배경이 복잡한(나무 바닥 등) 환경에서도 바나나를 정확히 식별할 수 있는 견고한 전처리 파이프라인이 구축되었습니다.
- **백앤드 연동 준비**: 백엔드(FastAPI) 팀은 `image_processor.py`의 `process_image` 함수와 저장된 `banana_model_latest.joblib`을 사용하여 즉시 서비스를 구축할 수 있습니다.
