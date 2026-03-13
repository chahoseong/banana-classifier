import io
import os
import cv2
import time
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import sys
from pathlib import Path

# Set up paths
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent

# Add paths to sys.path
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.inference import BananaGatekeeper
import joblib
from ml.src.preprocessing.image_processor import process_image

# Global instances
gatekeeper = None
ripeness_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gatekeeper, ripeness_model
    print("Initializing Models...")
    
    # 1. Gatekeeper (YOLO)
    gatekeeper = BananaGatekeeper(model_name='yolo26n.pt')
    
    # 2. Ripeness Model (scikit-learn)
    model_path = project_root / 'ml' / 'artifacts' / 'banana_model_latest.joblib'
    if model_path.exists():
        ripeness_model = joblib.load(str(model_path))
        print(f"Ripeness model loaded from {model_path}")
    else:
        print(f"Warning: Ripeness model not found at {model_path}")
        
    yield
    gatekeeper = None
    ripeness_model = None

app = FastAPI(title="Banana Ripe Checker API", lifespan=lifespan)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    print(f"Request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {duration:.3f}s")
    return response

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Banana Ripe Checker API is running"}


class PredictionResponse(BaseModel):
    is_banana: bool
    status: Optional[str]
    confidence: float
    message: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_ripeness(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_time = time.time()
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
            
        h, w = image_bgr.shape[:2]
        if h != 300 or w != 300:
            image_bgr = cv2.resize(image_bgr, (300, 300))
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

    if gatekeeper is None:
        raise HTTPException(status_code=500, detail="Gatekeeper is not loaded.")

    # 1. is_banana check using the Gatekeeper KNN model
    try:
        # predict returns boolean is_banana and a confidence score
        is_banana, confidence = gatekeeper.predict(image_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gatekeeper error: {e}")

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    if latency_ms > 300:
        print(f"Warning: Latency {(latency_ms):.2f}ms exceeded budget")

    # 3. Formulate the response according to FE guidelines
    if not is_banana:
        return PredictionResponse(
            is_banana=False,
            status=None,
            confidence=0.0, 
            message="바나나를 카메라 영역에 비춰주세요."
        )
    else:
        # Start ripeness classification
        status = "checking"
        prediction_message = "바나나 인식 완료! 상태 분석 중..."
        
        if ripeness_model is not None:
            try:
                # 1. Image processing (Masking + Feature Extraction)
                # Note: Ripeness model expects 6 features (Mean HSV, Std HSV)
                features = process_image(image_bgr)
                
                # 2. Predict status
                # Prediction returns the class name string directly if classes were provided to sklearn
                pred_status = ripeness_model.predict(features.reshape(1, -1))[0]
                status = str(pred_status)
                
                # 3. Get confidence (probability)
                if hasattr(ripeness_model, "predict_proba"):
                    probs = ripeness_model.predict_proba(features.reshape(1, -1))[0]
                    confidence = float(np.max(probs))
                
                prediction_message = f"숙성도 분석 완료: {status}"
            except Exception as e:
                print(f"Ripeness prediction error: {e}")
                prediction_message = "인식은 되었으나 숙성도 분석 중 오류가 발생했습니다."

        return PredictionResponse(
            is_banana=True,
            status=status,
            confidence=confidence,
            message=prediction_message
        )

if __name__ == "__main__":
    import uvicorn
    # Change from "backend.main:app" to "main:app" because we run from the backend folder
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
