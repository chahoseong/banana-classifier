import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class BananaGatekeeper:
    """
    Inference class for the is_banana YOLO model.
    Uses Ultralytics YOLO (COCO) to detect if a banana is present in the frame.
    """
    def __init__(self, model_name='yolo26n.pt'):
        # BANANA_ID = 46 in COCO dataset
        self.BANANA_ID = 46
        print(f"Initializing BananaGatekeeper with YOLO ({model_name})...")
        try:
            # This will automatically download the model if not found
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def predict(self, image_bgr: np.ndarray) -> tuple:
        """
        Predicts if the image contains a banana.
        Returns:
            is_banana (bool): True if at least one banana is detected.
            confidence (float): The highest confidence score among detected bananas.
        """
        if self.model is None or image_bgr is None:
            return False, 0.0
            
        try:
            # Inference (only looking for bananas, class 46)
            results = self.model(image_bgr, classes=[self.BANANA_ID], verbose=False)
            
            # results[0].boxes contains detection boxes
            if len(results[0].boxes) > 0:
                # Get the highest confidence score
                confidences = results[0].boxes.conf.tolist()
                max_confidence = float(max(confidences))
                return True, max_confidence
            else:
                return False, 0.0
                
        except Exception as e:
            print(f"Inference error: {e}")
            return False, 0.0
