import os
import cv2
import time
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Setup Paths relative to backend
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent
data_dir = project_root / 'dataset' / 'binary_check'
models_dir = backend_dir / 'models' / 'saved'
models_dir.mkdir(parents=True, exist_ok=True)

def extract_gatekeeper_features(image_bgr):
    """
    Extracts features for the KNN Gatekeeper:
    - Mean H, Mean S, Mean V
    - Yellow pixel ratio
    - Brown pixel ratio
    """
    if image_bgr is None:
        return None
        
    # Resize to 300x300 if not already (safeguard)
    h, w = image_bgr.shape[:2]
    if h != 300 or w != 300:
        image_bgr = cv2.resize(image_bgr, (300, 300))
        
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Calculate Mean H, S, V (excluding black padding: V > 0)
    # Actually, padding is [0, 0, 0] in BGR -> [0, 0, 0] in HSV. So V > 0 works.
    valid_pixels = image_hsv[image_hsv[:, :, 2] > 0]
    
    if len(valid_pixels) == 0:
        return np.zeros(5)
        
    mean_hsv = np.mean(valid_pixels, axis=0)
    
    # Calculate Yellow Ratio
    # H: 10-40, S: 50-255, V: 50-255
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    yellow_pixel_count = np.count_nonzero(yellow_mask)
    yellow_ratio = yellow_pixel_count / len(valid_pixels)
    
    # Calculate Brown Ratio
    # H: 0-20, S: 20-255, V: 20-150
    lower_brown = np.array([0, 20, 20])
    upper_brown = np.array([20, 255, 150])
    brown_mask = cv2.inRange(image_hsv, lower_brown, upper_brown)
    brown_pixel_count = np.count_nonzero(brown_mask)
    brown_ratio = brown_pixel_count / len(valid_pixels)
    
    # Return 5 features: [Mean_H, Mean_S, Mean_V, Yellow_Ratio, Brown_Ratio]
    features = np.array([
        mean_hsv[0], mean_hsv[1], mean_hsv[2], 
        yellow_ratio, brown_ratio
    ])
    return features

def load_data():
    X = []
    y = []
    
    print("Loading banana dataset...")
    banana_path = data_dir / 'banana'
    if banana_path.exists():
        for img_file in banana_path.glob('*.*'):
            img = cv2.imread(str(img_file))
            feat = extract_gatekeeper_features(img)
            if feat is not None:
                X.append(feat)
                y.append(1) # 1 for True (banana)
                
    print(f"Loaded {len(X)} banana images.")
    
    print("Loading not_banana dataset...")
    not_banana_path = data_dir / 'not_banana'
    not_bn_count = 0
    if not_banana_path.exists():
        for img_file in not_banana_path.glob('*.*'):
            img = cv2.imread(str(img_file))
            feat = extract_gatekeeper_features(img)
            if feat is not None:
                X.append(feat)
                y.append(0) # 0 for False (not banana)
                not_bn_count += 1
                
    print(f"Loaded {not_bn_count} not_banana images.")
    
    return np.array(X), np.array(y)

def train_and_save():
    X, y = load_data()
    
    if len(X) == 0:
        print("No data found to train model!")
        return
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN
    # Using small K and simple weights to keep latency < 30ms for preprocessing/inference
    print("Training KNN Model (K=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    
    start_time = time.time()
    knn.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Evaluate
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred) # TN, FP / FN, TP
    
    print("\n" + "="*40)
    print("          MODEL PERFORMANCE          ")
    print("="*40)
    print(f"Accuracy: {acc*100:.2f}%\n")
    print("Confusion Matrix:")
    print("                 Predicted Not-Banana | Predicted Banana")
    print(f"Actual Not-Banana:        {cm[0][0]:<14} | {cm[0][1]:<14}")
    print(f"Actual Banana    :        {cm[1][0]:<14} | {cm[1][1]:<14}\n")
    print(classification_report(y_test, y_pred, target_names=['Not Banana', 'Banana']))
    print("="*40)
    
    # Save Model and Scaler
    model_path = models_dir / 'banana_detector_model.pkl'
    scaler_path = models_dir / 'banana_detector_scaler.pkl'
    
    joblib.dump(knn, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

if __name__ == "__main__":
    train_and_save()
