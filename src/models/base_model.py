from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    모든 바나나 숙성도 분류 모델의 추상 기본 클래스입니다.
    새로운 모델을 추가할 때 이 클래스를 상속받아 구현해야 합니다.
    """

    @abstractmethod
    def load(self, model_path: str):
        """
        모델 파일을 로드합니다.
        
        Args:
            model_path (str): 모델 파일의 경로
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """
        이미지를 입력받아 숙성도를 예측합니다.
        
        Args:
            image (np.ndarray): OpenCV 형식(BGR) 또는 RGB 형식의 이미지 배열
            
        Returns:
            tuple[str, float]: (예측된 카테고리, 신뢰도/확률)
                카테고리는 'unripe', 'ripe', 'overripe', 'dispose' 중 하나여야 합니다.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        모델의 이름을 반환합니다.
        """
        pass
