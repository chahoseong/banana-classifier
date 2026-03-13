import numpy as np
from src.models.base_model import BaseModel
from src.models.ml_baseline import MLBaselineModel

class DummyModel(BaseModel):
    """
    초기 개발 및 UI 테스트를 위한 더미 모델입니다.
    무작위로 숙성도 결과를 반환합니다.
    """
    def __init__(self):
        self.categories = ['unripe', 'ripe', 'overripe', 'dispose']

    def load(self, model_path: str):
        print(f"Dummy model 'loaded' from {model_path}")

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        # 실제 모델이 없으므로 랜덤 결과 반환
        import random
        category = random.choice(self.categories)
        confidence = random.uniform(0.7, 0.99)
        return category, confidence

    @property
    def name(self) -> str:
        return "Dummy ML Model (Random)"

class InferenceEngine:
    """
    프론트엔드에서 요청하는 추론 작업을 관리하는 엔진 클래스입니다.
    """
    def __init__(self):
        self.model = None

    def set_model(self, model: BaseModel, model_path: str = None):
        """
        사용할 모델을 설정하고 로드합니다.
        """
        self.model = model
        if model_path:
            self.model.load(model_path)

    def infer(self, image: np.ndarray) -> tuple[str, float]:
        """
        입력 이미지에 대해 추론을 수행합니다.
        """
        if self.model is None:
            raise ValueError("모델이 설정되지 않았습니다.")
        
        # TODO: 이미지 전처리 로직 추가 (src.preprocessing 활용)
        return self.model.predict(image)
