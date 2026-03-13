import os
import warnings

import joblib
import numpy as np

from ml.inference.base_model import BaseModel
from ml.inference.feature_extractor import get_knn_features


class MLBaselineModel(BaseModel):
    """kNN baseline model wrapper for the reorganized workspace."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self._name = 'kNN Baseline Model'

    def load(self, model_dir: str):
        model_path = os.path.join(model_dir, 'knn_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f'Model artifacts not found: {model_dir}')

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        if self.model is None or self.scaler is None:
            raise ValueError('Model is not loaded. Call load() first.')

        features = get_knn_features(image)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scaled_features = self.scaler.transform(features)

        category = self.model.predict(scaled_features)[0]

        try:
            probs = self.model.predict_proba(scaled_features)[0]
            confidence = float(np.max(probs))
        except (AttributeError, ValueError):
            confidence = 1.0

        return category, confidence

    @property
    def name(self) -> str:
        return self._name
