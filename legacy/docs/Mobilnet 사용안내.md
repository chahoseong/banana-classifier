# 🍌 MobileNet V2 모델 사용 안내

> ⚠️ 현재 1 epoch만 학습한 모델입니다 (Val Accuracy ~65%). 기능 검증용으로 사용해 주세요.

## 1. 모델 파일 경로

```
weights/mobilenet_v2_ep1_acc65_0312.pth
```

## 2. 인터페이스 규격 (협업.md 기준)

| 항목 | 값 |
|---|---|
| 입력 | `(1, 3, 224, 224)` Tensor (RGB, ImageNet 정규화) |
| 출력 | `(1, 4)` Logits → softmax로 확률 변환 |
| 클래스 매핑 | `{0: 'Dispose', 1: 'Overripe', 2: 'Ripe', 3: 'Unripe'}` |

## 3. 모델 로드 + 추론 코드

```python
import torch
from PIL import Image
from data.augment import get_val_transforms
from models.custom_dl import CustomDLModel

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('weights/mobilenet_v2_ep1_acc65_0312.pth',
                        map_location=device, weights_only=False)

model = CustomDLModel(model_name=checkpoint['model_name'],
                      num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 클래스 매핑
idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
# {0: 'Dispose', 1: 'Overripe', 2: 'Ripe', 3: 'Unripe'}

# 단일 이미지 추론
image = Image.open('이미지경로.jpg').convert('RGB')
input_tensor = get_val_transforms()(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)

confidence, pred_idx = probs.max(1)
print(f"결과: {idx_to_class[pred_idx.item()]}, 신뢰도: {confidence.item()*100:.1f}%")
```

## 4. CLI로 빠른 테스트

```bash
.venv\Scripts\python.exe main.py --mode infer --weights weights/mobilenet_v2_ep1_acc65_0312.pth --image dataset/processed/test/Ripe/[파일명].jpg
```
