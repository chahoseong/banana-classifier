# banana-classifier

변경된 PRD에 맞춰 저장소 구조를 정비하는 마이그레이션을 진행 중입니다.

현재 작업 기준은 다음과 같습니다.

- `frontend/`: 새 프론트엔드 작업 공간
- `backend/`: 새 백엔드 작업 공간
- `ml/`: 학습 스크립트, 공용 추론 코드, 모델 아티팩트 작업 공간
- `dataset/`: 공용 데이터셋 위치
- `legacy/`: 기존 Streamlit 앱, 딥러닝 실험 코드, 이전 문서 보관 위치
- `docs/`: 현재 기준 구조 문서 위치

문서 시작점:

- `docs/repo-restructure-plan.md`
- `docs/migration-sequence.md`
- `docs/dependency-policy.md`

현재 정리된 주요 자산:

- 기존 Streamlit 앱: `legacy/streamlit_app/`
- 기존 딥러닝 실행 코드: `legacy/dl_pipeline/`
- 이전 문서: `legacy/docs/`
- kNN 학습 스크립트: `ml/training/train_knn.py`
- 공용 ML 추론 코드: `ml/inference/`
- 학습 산출물: `ml/artifacts/`

의존성 기준:

- 백엔드: `backend/requirements.txt`
- ML: `ml/requirements.txt`
- 레거시 경로: 루트 `requirements.txt`

공통 정책:

- 클래스는 `unripe`, `ripe`, `overripe`, `dispose` 4개를 유지합니다.
- 입력 해상도는 새 PRD 기준으로 `300x300`을 사용합니다.
- 기존 `src/` 코드는 즉시 삭제하지 않고, 새 구현이 안정화될 때까지 참고 자산으로 유지합니다.
