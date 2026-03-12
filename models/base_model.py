import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    모든 모델(DL, ML 등)에서 공통으로 상속받아 구현해야 하는 기본 인터페이스.
    협업.md 규격에 따라 BxCxHxW Tensor를 입력받고, 확률/로짓 배열을 반환해야 함.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.class_mapping = {
            0: 'unripe',
            1: 'ripe',
            2: 'overripe',
            3: 'dispose'
        }

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 형태 (B, C, H, W)
        Returns:
            torch.Tensor: 확률값(Softmax) 또는 로짓(Logit) 스코어 (B, 4)
        """
        raise NotImplementedError("서브클래스에서 추론 로직을 구현해야 합니다.")
