from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_metrics(y_true, y_pred, average='macro'):
    """
    Accuracy 외의 데이터 불균형 고려 지표 산출
    """
    f1 = f1_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    return {'F1': f1, 'Precision': precision, 'Recall': recall}

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    혼동 행렬 시각화
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()
