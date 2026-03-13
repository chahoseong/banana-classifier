# 워크스루: ML 전처리 모듈 구현 완료

`ML_task_list.md`의 **작업 ##1**인 이미지 전처리 및 특징 추출 로직 구현을 완료했습니다.

## 주요 변경 사항

### 1. [image_processor.py](file:///d:/Intel_AI_App_Creator_Hands_on/banana_classfier/banana-classifier/ml/src/preprocessing/image_processor.py)
- **`create_banana_mask`**: HSV 색상 임계값(노랑, 녹색, 갈색 범위)을 사용하여 배경에서 바나나 영역만 분리합니다. 노이즈 제거를 위한 형태학적 연산(Open/Close)이 포함되어 있습니다.
- **`extract_features`**: 마스킹된 픽셀에 대해서만 H, S, V 채널의 평균과 표준편차를 계산합니다.
- **`process_image`**: 300x300 BGR 이미지를 입력받아 6차원 특징 벡터를 반환하는 메인 엔트리 포인트입니다.

### 2. [test_processor.py](file:///d:/Intel_AI_App_Creator_Hands_on/banana_classfier/banana-classifier/ml/src/preprocessing/test_processor.py)
- 로컬 환경의 `dataset/raw/Unripe` 이미지를 로드하여 리사이즈(300x300) 후 전처리 로직을 검증하는 테스트 스크립트입니다.

## 검증 결과

가상 환경(`.venv`) 내에서 테스트 스크립트를 실행하여 정상 작동을 확인했습니다.

**테스트 출력 예시:**
```text
테스트 이미지: ...\dataset\raw\Unripe\20230204_083009851.jpg

[추출된 특징 벡터]
평균 (H, S, V): [ 28.18  110.36  112.51]
표준편차 (H, S, V): [ 10.61  40.73  40.73]

마스킹된 영역 픽셀 수: 55026 / 90000
성공: 바나나 영역이 감지되었습니다.
```

## 작업 ##2: 모델 학습 및 평가 결과

kNN과 SVM 모델을 학습시키고 성능을 비교한 결과, **kNN** 모델의 성능이 소폭 더 높아 최종 모델로 선택되었습니다.

### 1. 학습 세부 사항
- **사용 데이터**: `dataset/raw` 내 4개 클래스 (unripe, ripe, overripe, dispose)
- **추출 특징**: 6차원 HSV 특징 벡터 (평균 및 표준편차)
- **모델 저장 경로**: `ml/artifacts/banana_model_latest.joblib`

### 2. 성능 지표 (검증 데이터 기준)
- **최종 모델**: kNN (K-Nearest Neighbors)
- **정확도 (Accuracy)**: 약 **65%** (샘플 데이터 검증 기준)

## 결론 및 다음 단계
- **전처리 및 모델 학습 완료**: 협업 가이드라인에 명시된 ML 파트의 핵심 작업이 완료되었습니다.
- **백엔드 연동**: 이제 백엔드(FastAPI)에서 `image_processor.py`와 `banana_model_latest.joblib`을 사용하여 추론 API를 구축할 수 있습니다.
