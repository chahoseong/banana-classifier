# ML_task_list.md

본 문서는 `BE-ML 협업 가이드라인.md`를 바탕으로 ML 담당자가 수행해야 할 작업 목록을 정리한 체크리스트입니다.

## 1. 전처리 및 특징 추출 (Preprocessing & Feature Extraction)
- [ ] `ml/src/preprocessing/image_processor.py` 구현
    - [ ] 바나나 영역 마스킹(Masking) 로직 구현
    - [ ] HSV 변환 및 특징 추출 로직 구현
- [ ] 백엔드(FastAPI)와의 연동 테스트 (입력: 300x300 BGR Numpy Array)

## 2. 모델 개발 및 배포 (Model Development & Deployment)
- [ ] kNN (K-Nearest Neighbors) 및 SVM (Support Vector Machine) 기반 분류 모델 학습 (4개 클래스: `unripe`, `ripe`, `overripe`, `dispose`)
- [ ] 모델 아티팩트 저장: `ml/artifacts/banana_model_latest.joblib` 경로 고정
- [ ] 추론 인터페이스 제공
    - [ ] 상태 클래스(문자열)와 확신도(0.0~1.0)를 반환하는 함수 구현

## 3. 성능 최적화 (Optimization)
- [ ] 전체 ML 파이프라인 지연시간 250ms 이내 달성 확인
    - Target: ROI 추출 + 특징 변환 + K-NN/SVM 추론 시간 전체

## 4. 데이터 선순환 (Data Feedback Loop)
- [ ] `dataset/raw/low_confidence/` 경로에 적재된 저확신도 이미지 주기적 확인
- [ ] 오답 노트 라벨링 및 재학습(Retraining) 파이프라인 수행

## 5. 의존성 관리 (Dependency Management)
- [ ] `ml/requirements.txt` 내 주요 라이브러리 버전 확정
    - 필수 포함: `scikit-learn`, `opencv-python-headless`, `numpy` 등
