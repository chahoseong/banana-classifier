import io
import os
import cv2
import time
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
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
from models.cnn_model import BananaCNNModel
import torch
from torchvision import transforms
from PIL import Image
from fastapi import Form

# Global instances
gatekeeper = None
ripeness_model_baseline = None
ripeness_model_cnn = None

# CNN Preprocessing Transform
cnn_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gatekeeper, ripeness_model_baseline, ripeness_model_cnn
    print("Initializing Models...")
    
    # 1. Gatekeeper (YOLO)
    gatekeeper = BananaGatekeeper(model_name='yolo26n.pt')
    
    # 2. Baseline Model (KNN)
    baseline_path = project_root / 'ml' / 'artifacts' / 'banana_model_latest.joblib'
    if baseline_path.exists():
        ripeness_model_baseline = joblib.load(str(baseline_path))
        print(f"Baseline model loaded from {baseline_path}")
    
    # 3. CNN Model (MobileNetV2)
    cnn_path = project_root / 'ml' / 'artifacts' / 'banana_cnn_v1.pth'
    if cnn_path.exists():
        try:
            ripeness_model_cnn = BananaCNNModel(num_classes=4)
            checkpoint = torch.load(str(cnn_path), map_location='cpu', weights_only=False)
            # Handle both raw state_dict and checkpoint dict
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            ripeness_model_cnn.load_state_dict(state_dict)
            ripeness_model_cnn.eval()
            print(f"CNN model loaded from {cnn_path}")
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            
    yield
    gatekeeper = None
    ripeness_model_baseline = None
    ripeness_model_cnn = None

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
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    # Get model info if it was set during the request
    model_id = getattr(request.state, "model_id", "SYSTEM")
        
    print(f"[{model_id}] {request.method} {request.url.path} - Status: {response.status_code} - {duration:.3f}s", flush=True)
    return response

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Banana Ripe Checker API is running"}


@app.get("/models")
async def get_models():
    return [
        {
            "id": "KNN",
            "name": "Baseline Model (KNN)",
            "description": "이미지 색상 분포를 기반으로 한 기본 분석 모델",
            "type": "classic_ml"
        },
        {
            "id": "CNN",
            "name": "MobileNetV2 (CNN)",
            "description": "딥러닝 알고리즘을 사용한 고정밀 분석 모델",
            "type": "deep_learning"
        }
    ]

class PredictionResponse(BaseModel):
    is_banana: bool
    status: Optional[str]
    confidence: float
    message: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_ripeness(
    request: Request,
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    model_id: str = Form("KNN")
):
    # Store model_id in request state for logging middleware
    request.state.model_id = model_id
    start_time = time.time()
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
            
        # Keep original for potential reuse, but resize for inference
        h, w = image_bgr.shape[:2]
        if h != 300 or w != 300:
            image_bgr = cv2.resize(image_bgr, (300, 300))
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")

    if gatekeeper is None:
        raise HTTPException(status_code=500, detail="Gatekeeper is not loaded.")

    # 1. is_banana check using YOLO
    try:
        is_banana, confidence = gatekeeper.predict(image_bgr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gatekeeper error: {e}")

    if not is_banana:
        return PredictionResponse(
            is_banana=False,
            status=None,
            confidence=0.0, 
            message="바나나를 카메라 영역에 비춰주세요."
        )
    
    # 2. Ripeness classification based on model_id
    status = "checking"
    prediction_message = f"바나나 인식 완료 ({model_id})! 상태 분석 중..."
    
    try:
        if model_id in ["CNN", "cnn_mobilenet"]:
            if ripeness_model_cnn is not None:
                # Convert BGR to RGB PIL Image for transforms
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(image_rgb)
                input_tensor = cnn_transforms(pil_img).unsqueeze(0)
                
                status, confidence = ripeness_model_cnn.predict(input_tensor)
                prediction_message = f"CNN 분석 완료: {status}"
            else:
                prediction_message = "CNN 모델이 로드되지 않았습니다."
        else: # Default or "baseline"
            if ripeness_model_baseline is not None:
                features = process_image(image_bgr)
                pred_status = ripeness_model_baseline.predict(features.reshape(1, -1))[0]
                status = str(pred_status)
                
                if hasattr(ripeness_model_baseline, "predict_proba"):
                    probs = ripeness_model_baseline.predict_proba(features.reshape(1, -1))[0]
                    confidence = float(np.max(probs))
                
                prediction_message = f"숙성도 분석 완료: {status}"
            else:
                prediction_message = "베이스라인 모델이 로드되지 않았습니다."
                
    except Exception as e:
        print(f"Prediction error ({model_id}): {e}")
        prediction_message = f"인식은 되었으나 {model_id} 분석 중 오류가 발생했습니다."

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
