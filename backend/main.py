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
from pydantic import BaseModel
from typing import Optional

# Set up paths
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent
# Import BananaGatekeeper
import sys
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))
from src.inference import BananaGatekeeper

# Global instance for the gatekeeper
gatekeeper = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gatekeeper
    print("Initializing BananaGatekeeper...")
    gatekeeper = BananaGatekeeper()
    yield
    gatekeeper = None

app = FastAPI(title="Banana Ripe Checker API", lifespan=lifespan)

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
        # Temporary response for the actual ripeness model integration later
        # is_banana equals True
        return PredictionResponse(
            is_banana=True,
            status="checking",
            confidence=0.8,
            message="바나나 인식 완료! 상태 분석 중..."
        )

if __name__ == "__main__":
    import uvicorn
    # Change from "backend.main:app" to "main:app" because we run from the backend folder
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
