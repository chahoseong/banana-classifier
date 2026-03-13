# Banana Ripe Checker PRD

## 1. 프로젝트 개요

**프로젝트 명**
Banana Ripe Checker (실시간 바나나 숙성도 판별기)

**프로젝트 목적**
사용자가 카메라에 바나나를 비추면 AI가 숙성 상태를 자동으로 판별하여 사용자에게 시각적으로 알려주는 웹 애플리케이션을 개발한다.

이 프로젝트는 **Computer Vision 기반 AI 웹 애플리케이션 개발 연습**을 목적으로 한다.

**타겟 사용자**

* 신선한 과일을 구매하려는 일반 소비자
* 식재료 관리가 필요한 요리사
* 가정에서 과일 상태를 확인하려는 사용자

---

# 2. 주요 기능 (User Features)

## 2.1 실시간 카메라 스트리밍

* 웹 브라우저에서 카메라 접근 권한을 요청한다.
* 노트북 또는 연결된 카메라의 영상을 실시간으로 화면에 표시한다.

사용 기술

* `navigator.mediaDevices.getUserMedia`

---

## 2.2 자동 숙성도 판별

* 별도의 촬영 버튼 없이 자동으로 분석한다.
* 일정 시간 간격으로 현재 프레임을 캡처한다.

분석 주기

```
0.5 ~ 1초
```

---

## 2.3 시각적 피드백 제공

AI 판별 결과에 따라 화면에 다음 정보를 표시한다.

* 숙성 상태 텍스트
* 색상 가이드라인

예시

```
ACCESS GRANTED - RIPE
```

또는

```
UNRIPE BANANA
OVER-RIPE BANANA
```

---

## 2.4 숙성 단계 분류

| 상태           | 설명                      |
| ------------ | ----------------------- |
| **Unripe**   | 초록색, 아직 익지 않은 상태        |
| **Ripe**     | 노란색, 바로 먹기 좋은 상태        |
| **Overripe** | 갈색 반점이 많은 상태, 베이킹 등에 적합 |

---

# 3. 기술 요구사항 (Technical Requirements)

## 3.1 프론트엔드 (React)

### 카메라 제어

브라우저 API를 사용하여 카메라 영상 스트림을 가져온다.

사용 API

```
navigator.mediaDevices.getUserMedia
```

---

### 프레임 캡처

Canvas API를 사용하여 영상 프레임을 캡처한다.

이미지 전처리

```
resolution: 300 x 300
```

목적

* 처리 속도 향상
* 학습 데이터와 동일한 해상도 유지

---

### 이미지 전송

캡처된 이미지를 일정 주기로 백엔드로 전송한다.

전송 방식

```
FormData
```

전송 주기 및 로직 (Throttle)

* 무조건적인 고정 주기 전송이 아닌, **이전 요청에 대한 서버의 응답을 수신한 후 다음 프레임을 전송**하여 네트워크 큐가 쌓이는 것을 방지한다.
* 최소 대기 시간(예: 500ms)을 두어 서버 부하를 조절하는 로직을 프론트엔드 핵심 설계에 포함한다.

---

## 3.2 백엔드 (FastAPI)

### 이미지 업로드 API

FastAPI 엔드포인트를 통해 이미지를 수신한다.

예시

```
POST /predict
```

입력 데이터

```
UploadFile
```

---

### 이미지 처리

업로드된 이미지를 OpenCV 형식으로 변환한다.

```
OpenCV (cv2)
```

---

### 특징 추출 (Feature Extraction)

다음 정보를 추출한다.

1. 바나나 영역 마스킹
2. 평균 색상 계산

계산 색상 공간

```
HSV (조명 밝기에 덜 민감하여 BGR보다 바나나 색상 분류에 압도적으로 유리함)
```

추출 feature 예시

```
mean_H
mean_S
mean_V
yellow_ratio
brown_ratio
```

---

### 추론 (Inference)

추출된 feature를 기반으로 ML 모델을 통해 숙성도를 예측한다.

사용 모델

```
kNN (K-Nearest Neighbors)
또는
SVM (Support Vector Machine)
```

출력 예시

```json
{
  "status": "Ripe",
  "confidence": 0.87
}
```

---

# 4. AI 모델링 (Machine Learning)

## 4.1 데이터셋

직접 촬영 및 라벨링한 바나나 이미지를 사용한다.

데이터 클래스

```
Unripe
Ripe
Overripe
```

권장 데이터 수

```
각 클래스 100~300 이미지
```

다양한 환경에서 촬영

* 밝은 조명
* 어두운 조명
* 다양한 배경

---

## 4.2 Feature Engineering

모델 입력 feature

```
HSV 평균값
HSV 히스토그램
갈색 픽셀 비율
노란색 픽셀 비율
```

---

## 4.3 모델 알고리즘

사용 가능한 알고리즘

### kNN

* 구현이 간단
* 데이터 분포 기반 분류

### SVM

* 작은 feature 공간에서 좋은 성능
* 명확한 decision boundary 생성

---

## 4.4 입력 이미지 규격

모든 이미지 입력은 동일한 해상도로 처리한다.

```
300 x 300
```

목적

* 처리 속도 향상
* 모델 일관성 유지

---

# 5. 서비스 흐름 (User Flow)

```
사용자 접속
        │
        ▼
카메라 권한 승인
        │
        ▼
React 앱이 실시간 영상 출력
        │
        ▼
Canvas에서 프레임 캡처 (300x300)
        │
        ▼
FastAPI 서버로 이미지 전송
        │
        ▼
OpenCV 이미지 처리
        │
        ▼
Feature Extraction
        │
        ▼
ML 모델 추론
        │
        ▼
JSON 결과 반환
        │
        ▼
React UI 결과 표시
```

---

# 6. UI 피드백

예시 화면 표시

| 상태       | 화면 표시   |
| -------- | ------- |
| Unripe   | 녹색 테두리  |
| Ripe     | 노란색 테두리 |
| Overripe | 갈색 테두리  |

텍스트 예시

```
UNRIPE BANANA
RIPE BANANA
OVER-RIPE BANANA
```

---

# 7. 성공 지표 (Success Metrics)

## 정확도

```
Accuracy ≥ 85%
```

테스트 데이터 기준

---

## 응답 속도

이미지 업로드부터 결과 반환까지

```
Latency ≤ 500 ms
```

---

## 안정성

다음 환경에서도 일정 수준의 성능 유지

* 밝은 조명
* 어두운 조명
* 다양한 배경

---

# 8. 프로젝트 목적 (Learning Objectives)

이 프로젝트를 통해 다음 기술을 학습한다.

* Computer Vision 기초
* Feature Engineering
* Classical Machine Learning
* React 기반 카메라 처리
* FastAPI 기반 AI API 구축
* 실시간 AI 웹 애플리케이션 개발

---

# 9. 기술 스택

| 영역               | 기술           |
| ---------------- | ------------ |
| Frontend         | React        |
| Backend          | FastAPI      |
| Computer Vision  | OpenCV       |
| Machine Learning | scikit-learn |
| Image Processing | NumPy        |
| Visualization    | HTML5 Canvas |

# 10. 향후 고도화 계획

## 1. 모델 성능 개선 (Model Accuracy & Robustness)

단순한 평균 색상 추출만으로는 조명 변화나 복잡한 배경에서 한계가 올 수 있습니다.

* **'None' (배경) 클래스 추가**: 현재 모델은 빈 화면이나 바나나가 아닌 물체도 억지로 분류하려드는 문제가 생길 수 있으므로, 향후 데이터셋에 '배경만 있는 사진'이나 '기타 물체'를 포함한 None 또는 Background 클래스를 추가하여 앱의 안정성을 높입니다.
* **HSV 색 공간 활용**: BGR 대신 조명(밝기)에 덜 민감한 **HSV(Hue, Saturation, Value)** 색 공간을 사용하여 '색상(Hue)' 정보에 더 집중합니다.
* **관심 영역(ROI) 자동 추출**: 이미지 전체의 평균을 내지 않고, **Contour Detection(윤곽선 감지)**을 통해 바나나만 따낸 뒤 그 안의 픽셀들만 분석합니다.
* **앙상블/다수결 로직**: 0.5초마다 들어오는 결과값들에 대해 **최빈값(Mode)** 필터를 적용하여, 연속적으로 "Ripe"가 3번 이상 나올 때만 화면을 갱신하도록 처리합니다.

---

## 2. 데이터 전송 효율성 개선 (Efficiency & Latency)

실시간 웹 앱에서 가장 병목이 생기는 지점은 네트워크 전송입니다.

* **이미지 압축 및 리사이징**: 캔버스에서 이미지를 뽑을 때 `image/jpeg` 포맷과 `quality` 옵션(예: 0.7)을 조절하여 파일 용량을 극적으로 줄입니다.
* **Throttle 기반 전송**: 무조건적인 `setInterval` 대신, **"백엔드 응답이 돌아온 후 다음 프레임을 전송"**하는 재귀적 호출 방식으로 변경하여 네트워크 큐(Queue)가 쌓이는 것을 방지합니다.
* **WebSockets 고려**: HTTP POST 방식 대신 **WebSocket**을 사용하면 오버헤드를 줄여 더 낮은 지연 시간으로 양방향 통신이 가능해집니다.

---

## 3. 데이터 로깅 및 재학습 파이프라인

* **데이터 자동 수집**: 모델이 확신이 낮은(Confidence Low) 결과나 오답으로 판명된 이미지를 별도 서버 폴더에 저장하여, 나중에 라벨링 후 **재학습(Retraining)**에 사용합니다.