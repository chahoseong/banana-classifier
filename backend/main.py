import io
import os
import cv2
import time
import uuid
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel

# Correctly import from ml directory
import sys
# Add project root to sys.path to allow imports from ml
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ml.src.preprocessing.image_processor import process_image

# Global variables to store the loaded model
MODEL_PATH = project_root / 'ml' / 'artifacts' / 'banana_model_latest.joblib'
LOW_CONFIDENCE_DIR = project_root / 'dataset' / 'raw' / 'low_confidence'

# Ensure low confidence dir exists
LOW_CONFIDENCE_DIR.mkdir(parents=True, exist_ok=True)

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Perform startup actions
    if not MODEL_PATH.exists():
        print(f"Warning: Model not found at {MODEL_PATH}")
    else:
        try:
            model = joblib.load(str(MODEL_PATH))
            print(f"Model successfully loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            
    yield
    # Perform shutdown actions (if any)
    model = None

app = FastAPI(title="Banana Ripe Checker API", lifespan=lifespan)

class PredictionResponse(BaseModel):
    is_banana: bool
    status: str
    confidence: float
    message: str

def save_low_confidence_image(image_bgr: np.ndarray, status: str):
    """Background task to save images with low confidence < 0.6"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:6]
    filename = f"{timestamp}_{unique_id}_pred_{status}.jpg"
    filepath = LOW_CONFIDENCE_DIR / filename
    try:
        cv2.imwrite(str(filepath), image_bgr)
        print(f"Saved low confidence image: {filepath}")
    except Exception as e:
        print(f"Failed to save low confidence image: {e}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_ripeness(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_time = time.time()
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        # Read image bytes
        contents = await file.read()
        
        # Decode image to numpy array (BGR)
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
            
        # Ensure image is exactly 300x300, pad or resize if necessary based on BE-ML agreements
        h, w = image_bgr.shape[:2]
        if h != 300 or w != 300:
            image_bgr = cv2.resize(image_bgr, (300, 300))
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

    # Extract features using ML team's processor
    try:
        features = process_image(image_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")
        
    # Check is_banana logic (Gatekeeper)
    # The image_processor returns an array of zeros if no pixels pass the banana mask
    if np.all(features == 0):
        return PredictionResponse(
            is_banana=False,
            status="unknown",
            confidence=0.0,
            message="No banana detected in the image."
        )
        
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on server.")
        
    # Inference
    try:
        # Reshape for single sample prediction
        features_reshaped = features.reshape(1, -1)
        
        # Depending on the model (KNN/SVM), predict and predict_proba
        prediction = model.predict(features_reshaped)[0]
        
        # Get confidence if model supports it (like KNeighborsClassifier or SVC with probability)
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_reshaped)[0]
            confidence = float(np.max(probabilities))
            
        # Format the status string smoothly (unripe, ripe, overripe, dispose)
        status_str = str(prediction).lower().strip()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
        
    # Low confidence logging logic
    if confidence < 0.6:
        background_tasks.add_task(save_low_confidence_image, image_bgr, status_str)
        
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    # Check latency budget (target < 250ms total ML inference theoretically)
    if latency_ms > 300:
        print(f"Warning: Latency budget exceeded ({latency_ms:.2f}ms)")
        
    return PredictionResponse(
        is_banana=True,
        status=status_str,
        confidence=confidence,
        message="Ripeness predicted successfully."
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
