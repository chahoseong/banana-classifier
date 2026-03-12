from .base_model import BaseModel

class MachineLearningBaseline(BaseModel):
    """
    팀원 1 담당: 전통적 머신러닝 방식의 베이스라인 추론기.
    """
    def __init__(self):
        super().__init__()
        # TODO: 로드할 학습된 ML 모델 (RandomForest, SVM 등) 객체 선언

    def forward(self, x):
        """
        텐서 입력을 받아 ML 방식의 특징(Feature) 추출 후 예측 결과를 텐서로 변환하여 반환
        """
        # TODO: 텐서 -> NumPy 배열 변환 -> 특징 1D 추출 -> scikit-learn 예측
        pass
