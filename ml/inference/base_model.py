from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Common interface for banana ripeness classifiers."""

    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> tuple[str, float]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
