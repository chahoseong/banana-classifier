import joblib
import numpy as np
import os
from src.models.base_model import BaseModel
from src.preprocessing.image_processor import get_knn_features

class MLBaselineModel(BaseModel):
    """
    팀원 1의 kNN 베이스라인 모델을 연동하는 클래스입니다.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self._name = "kNN Baseline Model"

    def load(self, model_dir: str):
        """
        모델 폴더 내의 knn_model.pkl과 scaler.pkl을 로드합니다.
        """
        model_path = os.path.join(model_dir, 'knn_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"모델 또는 스케일러 파일을 찾을 수 없습니다: {model_dir}")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"[{self._name}] 모델 및 스케일러 로드 완료.")

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """
        이미지에서 특징을 추출하고 kNN으로 예측합니다.
        
        Returns:
            tuple[str, float]: (예측 카테고리, 확률)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("모델이 로드되지 않았습니다. load()를 먼저 호출하세요.")
            
        # 1. 516차원 특징 추출 (image_processor 활용)
        features = get_knn_features(image)
        
        # 2. 스케일링 (피처 이름 경고 방지를 위해 values만 전달)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_features = self.scaler.transform(features)
        
        # 3. 예측
        category = self.model.predict(scaled_features)[0]
        
        # 4. 확률(신뢰도) 계산
        # kNN의 경우 가까운 이웃의 비율로 확률을 계산할 수 있음 (predict_proba 지원 시)
        try:
            probs = self.model.predict_proba(scaled_features)[0]
            confidence = np.max(probs)
        except (AttributeError, ValueError):
            # 확률을 지원하지 않는 모델의 경우 기본값 1.0 (또는 별도 계산 방식 적용)
            confidence = 1.0
            
        return category, confidence

    @property
    def name(self) -> str:
        return self._name
