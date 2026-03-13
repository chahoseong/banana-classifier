import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import sys

# 프로젝트 루트 및 모듈 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
import image_processor

def load_dataset(dataset_path):
    features_list = []
    labels_list = []
    
    # 클래스 매핑 (실제 폴더명 -> 가이드라인 스펙 소문자 명칭)
    class_mapping = {
        "Unripe": "unripe",
        "Ripe": "ripe",
        "Overripe": "overripe",
        "Dispose": "dispose"
    }
    
    for folder_name, label in class_mapping.items():
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"경고: {folder_path} 폴더를 찾을 수 없습니다.")
            continue
            
        print(f"클래스 '{label}' 데이터 로딩 중...")
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg'):
                file_path = os.path.join(folder_path, file_name)
                
                # 이미지 로드 및 리사이즈
                img = cv2.imread(file_path)
                if img is None:
                    continue
                img = cv2.resize(img, (300, 300))
                
                # 전처리 및 특징 추출
                features = image_processor.process_image(img)
                
                # 마스킹 실패(바나나 감지 안됨)한 데이터는 학습에서 제외하거나 0 벡터로 처리
                if np.sum(features) == 0:
                    continue
                
                features_list.append(features)
                labels_list.append(label)
                
    return np.array(features_list), np.array(labels_list)

def train_and_evaluate():
    dataset_path = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\dataset\raw"
    artifacts_dir = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\ml\artifacts"
    
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    # 1. 데이터셋 로드
    X, y = load_dataset(dataset_path)
    print(f"\n총 데이터 수: {len(X)}")
    
    if len(X) == 0:
        print("에러: 학습할 데이터가 없습니다.")
        return

    # 2. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. kNN 모델 학습 및 평가
    print("\n--- kNN 모델 학습 및 평가 ---")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, knn_preds):.4f}")
    print(classification_report(y_test, knn_preds))

    # 4. SVM 모델 학습 및 평가
    print("\n--- SVM 모델 학습 및 평가 ---")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, svm_preds):.4f}")
    print(classification_report(y_test, svm_preds))

    # 5. 성능이 더 좋은 모델 선택 및 저장
    knn_acc = accuracy_score(y_test, knn_preds)
    svm_acc = accuracy_score(y_test, svm_preds)
    
    best_model = svm if svm_acc >= knn_acc else knn
    model_name = "SVM" if svm_acc >= knn_acc else "kNN"
    
    save_path = os.path.join(artifacts_dir, "banana_model_latest.joblib")
    joblib.dump(best_model, save_path)
    print(f"\n최적 모델 저장 완료: {model_name} (Accuracy: {max(knn_acc, svm_acc):.4f}) -> {save_path}")

if __name__ == "__main__":
    train_and_evaluate()
