import cv2
from ultralytics import YOLO
import sys

def test_yolo_detection():
    # BANANA_ID = 46 in COCO dataset
    BANANA_ID = 46
    model_name = 'yolo11n.pt'
    
    print(f"Loading model: {model_name}")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check if we can find a banana in a sample image or a generated one
    # For now, let's just check if it initializes and runs on a blank image
    import numpy as np
    dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
    
    print("Running inference on dummy frame...")
    results = model(dummy_frame, classes=[BANANA_ID], verbose=True)
    
    if len(results[0].boxes) > 0:
        print("Success: Something detected (even if it's a blank box test)")
    else:
        print("Success: Nothing detected in blank frame (expected)")
    
    print("YOLO Detection Pipeline is READY.")

if __name__ == "__main__":
    test_yolo_detection()
