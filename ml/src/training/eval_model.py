import joblib
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import sys

# 프로젝트 루트 및 모듈 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
import image_processor

def final_eval():
    dataset_path = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\dataset\raw"
    model_path = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\ml\artifacts\banana_model_latest.joblib"
    
    model = joblib.load(model_path)
    
    features_list = []
    labels_list = []
    class_mapping = {"Unripe": "unripe", "Ripe": "ripe", "Overripe": "overripe", "Dispose": "dispose"}
    
    for folder_name, label in class_mapping.items():
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path): continue
        
        files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
        for file_name in files[:50]: # 샘플 50개씩만 확인
            img = cv2.imread(os.path.join(folder_path, file_name))
            if img is None: continue
            img = cv2.resize(img, (300, 300))
            features = image_processor.process_image(img)
            if np.sum(features) == 0: continue
            features_list.append(features)
            labels_list.append(label)
            
    X = np.array(features_list)
    y = np.array(labels_list)
    
    preds = model.predict(X)
    print(f"Final Validation Accuracy (Sample): {accuracy_score(y, preds):.4f}")
    print(classification_report(y, preds))

if __name__ == "__main__":
    final_eval()
