import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
try:
    from utils.metrics import calculate_metrics, plot_confusion_matrix
except ImportError:
    # utils 패키지 임포트 실패 대비 대체
    def calculate_metrics(*args, **kwargs): pass
    def plot_confusion_matrix(*args, **kwargs): pass

def train_and_optimize_knn(csv_path='ml/artifacts/features.csv'):
    """
    추출된 특징(features.csv) 데이터를 불러와 K-Nearest Neighbors(KNN) 모델을 학습하고 
    하이퍼파라미터(K값)를 최적화합니다.
    """
    print(f"'{csv_path}' 파일에서 데이터 로드 중...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} 파일을 찾을 수 없습니다. extract_features.py를 먼저 실행해주세요.")
        return
        
    print(f"원본 데이터 크기: {df.shape}")

    # 1. 필요 없는 컬럼 제외하고 Features(X)와 Target(y) 분리
    # filename, split 컬럼은 학습 특성에서 제외
    drop_cols = ['filename', 'split', 'label']
    
    # 예외처리: 컬럼이 부족한 경우 대비
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=existing_drop_cols)
    y = df['label']
    
    # 2. Train / Test Split 분할
    # extract_features.py에서 저장한 'split' 컬럼을 활용하거나, 새로 분할합니다.
    if 'split' in df.columns:
        print("기존에 분할된 split 정보를 바탕으로 데이터를 나눕니다.")
        train_idx = df[df['split'] == 'train'].index
        test_idx = df[df['split'] == 'test'].index
        # validation도 있지만 단순화를 위해 train/test로 봅니다.
        
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    else:
        print("split 컬럼이 없어 새로 8:2 분할을 수행합니다.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
    print(f"학습용 데이터 수: {len(X_train)}")
    print(f"테스트용 데이터 수: {len(X_test)}")

    # 3. 데이터 스케일링 (KNN은 거리 기반 알고리즘이므로 필수!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. KNN 하이퍼파라미터 튜닝 (Grid Search)
    print("\n[GridSearchCV] 최적의 K값을 찾는 중...")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],  # 튜닝할 K값 후보
        'weights': ['uniform', 'distance'],   # 거리 가중치 여부
        'metric': ['euclidean', 'manhattan']  # 거리 계산 단위
    }
    
    # 기본 분류기 생성
    knn_base = KNeighborsClassifier()
    
    # 5-Fold 교차 검증으로 최적 모델 찾기
    grid_search = GridSearchCV(
        estimator=knn_base, 
        param_grid=param_grid, 
        cv=5,                 # 폴드 개수
        scoring='accuracy',   # 평가 기준
        n_jobs=-1             # 모든 CPU 코어 사용
    )
    
    # 모델 학습 및 최적화
    grid_search.fit(X_train_scaled, y_train)

    # 최고 성능의 파라미터 및 모델
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(f"🔥 가장 높은 정확도를 보인 파라미터: {best_params}")
    print(f"📈 검증 Fold 최고 정확도(Validation Accuracy): {grid_search.best_score_:.4f}")

    # 5. 최종 평가(Test Set)
    print("\n--- 최적화된 모델로 테스트 세트 최종 평가 ---")
    y_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    
    from sklearn.metrics import f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    test_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"최종 테스트 정확도(Test Accuracy): {test_acc:.4f}")
    print(f"최종 테스트 F1-Score (Macro): {test_f1:.4f}")
    
    print("\n[상세 분류 보고서 - Classification Report]")
    print(classification_report(y_test, y_pred))
    
    # 6. Confusion Matrix 시각화 및 저장
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('KNN Baseline Confusion Matrix')
    
    plt.tight_layout()
    report_path = os.path.join('ml', 'artifacts', 'baseline_report.png')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    plt.savefig(report_path, dpi=300)
    print(f"\n[??] ?? ?? ??? ??? '{report_path}'? ???????.")
    
    # 7. 모델 및 스케일러 저장 (추가)
    import joblib
    save_path = os.path.join('ml', 'artifacts')
    os.makedirs(save_path, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(save_path, 'knn_model.pkl'))
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
    print(f"\n✅ 모델 및 스케일러가 '{save_path}'에 저장되었습니다.")
    
if __name__ == "__main__":
    train_and_optimize_knn()
