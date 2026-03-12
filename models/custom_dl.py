import torch
import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel

class CustomDLModel(BaseModel):
    """
    팀원 2 담당: 딥러닝 기반 이미지 분류 모델 (ResNet, MobileNet 등)
    """
    def __init__(self, num_classes=4):
        super().__init__()
        # 예시: ResNet18을 가져와서 마지막 레이어만 변경
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
