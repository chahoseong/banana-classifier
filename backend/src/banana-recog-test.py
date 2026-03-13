from ultralytics import YOLO
import cv2

# 1. 모델 로드 (바나나가 포함된 사전 학습 모델)
model = YOLO('yolo26n.pt')

# 2. 웹캠 연결
cap = cv2.VideoCapture(0)

# COCO 데이터셋에서 바na나의 클래스 번호는 46번입니다.
BANANA_ID = 46

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 모델 추론 (바나나 클래스만 필터링하여 성능 최적화)
    results = model(frame, classes=[BANANA_ID], verbose=False)

    # 바나나가 하나라도 감지되었는지 확인
    # results[0].boxes 데이터가 비어있지 않으면 바나나가 있는 것임
    if len(results[0].boxes) > 0:
        status = "Banana"
    else:
        status = "none..."

    # 터미널 출력
    print(f"\r현재 상태: {status}", end="")

    # (옵션) 화면에 상태 표시
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Banana Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()