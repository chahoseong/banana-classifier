import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel


SUPPORTED_MODELS = ['mobilenet_v2', 'resnet18', 'resnet50']


class CustomDLModel(BaseModel):
    """
    팀원 2 담당: 딥러닝 기반 이미지 분류 모델.
    PRD 제약 조건에 따라 여러 모델을 교체하여 사용할 수 있도록 구현.
    """
    def __init__(self, model_name='mobilenet_v2', num_classes=4):
        super().__init__()
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"지원하지 않는 모델: {model_name}. 선택 가능: {SUPPORTED_MODELS}")

        self.model_name = model_name

        if model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)

        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

        elif model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
