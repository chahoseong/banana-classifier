import os
import cv2
import joblib
import numpy as np
from pathlib import Path

class BananaGatekeeper:
    """
    Inference class for the is_banana KNN model.
    Checks if an image is a banana or not based on HSV and pixel ratios.
    """
    def __init__(self):
        backend_dir = Path(__file__).resolve().parent.parent
        models_dir = backend_dir / 'models' / 'saved'
        
        model_path = models_dir / 'banana_detector_model.pkl'
        scaler_path = models_dir / 'banana_detector_scaler.pkl'
        
        if not model_path.exists() or not scaler_path.exists():
            print(f"Warning: Gatekeeper Model ({model_path.name}) or Scaler not found!")
            self.model = None
            self.scaler = None
        else:
            self.model = joblib.load(str(model_path))
            self.scaler = joblib.load(str(scaler_path))

    def _extract_features(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Extract the exactly same 5 features used in training.
        """
        if image_bgr is None:
            return None
        
        h, w = image_bgr.shape[:2]
        if h != 300 or w != 300:
            image_bgr = cv2.resize(image_bgr, (300, 300))
            
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        valid_pixels = image_hsv[image_hsv[:, :, 2] > 0]
        
        if len(valid_pixels) == 0:
            return np.zeros(5)
            
        mean_hsv = np.mean(valid_pixels, axis=0)
        
        lower_yellow = np.array([10, 50, 50])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
        yellow_ratio = np.count_nonzero(yellow_mask) / len(valid_pixels)
        
        lower_brown = np.array([0, 20, 20])
        upper_brown = np.array([20, 255, 150])
        brown_mask = cv2.inRange(image_hsv, lower_brown, upper_brown)
        brown_ratio = np.count_nonzero(brown_mask) / len(valid_pixels)
        
        features = np.array([
            mean_hsv[0], mean_hsv[1], mean_hsv[2], 
            yellow_ratio, brown_ratio
        ])
        return features

    def predict(self, image_bgr: np.ndarray) -> tuple:
        """
        Predicts if the image is a banana.
        Returns:
            is_banana (bool): True if banana, False otherwise.
            confidence (float): The prediction probability/confidence.
        """
        if self.model is None or self.scaler is None:
            # Fallback if model failed to load
            return False, 0.0
            
        features = self._extract_features(image_bgr)
        if features is None or np.all(features == 0):
            return False, 0.0
            
        features_reshaped = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features_reshaped)
        
        prediction = self.model.predict(features_scaled)[0]
        
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = float(np.max(probabilities))
            
        is_banana = bool(prediction == 1)
        
        return is_banana, confidence
