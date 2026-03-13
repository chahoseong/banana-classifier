import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 프로젝트 루트 및 모듈 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'preprocessing'))
import image_processor

class BananaDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, use_mask=True):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.use_mask = use_mask
        self.class_to_idx = {"unripe": 0, "ripe": 1, "overripe": 2, "dispose": 3}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            image_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        image_bgr = cv2.resize(image_bgr, (224, 224))
        
        if self.use_mask:
            mask = image_processor.create_banana_mask(image_bgr)
            mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
            image_masked = (image_bgr * mask_3ch).astype(np.uint8)
            image_rgb = cv2.cvtColor(image_masked, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            image_rgb = self.transform(image_rgb)
        return image_rgb, self.class_to_idx[label]

def evaluate_model():
    dataset_path = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\dataset\raw"
    model_path = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\ml\artifacts\banana_cnn_v1.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 로드
    file_paths = []
    labels = []
    class_mapping = {"Unripe": "unripe", "Ripe": "ripe", "Overripe": "overripe", "Dispose": "dispose"}
    
    for folder_name, label in class_mapping.items():
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path): continue
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(folder_path, file_name))
                labels.append(label)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = BananaDataset(file_paths, labels, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 모델 로드
    model = models.efficientnet_b0()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("평가 중...")
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.numpy())
            
    # 결과 보고
    target_names = ["unripe", "ripe", "overripe", "dispose"]
    print("\n[CNN (EfficientNet-B0) Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Confusion Matrix 저장
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('CNN Confusion Matrix')
    cm_path = os.path.join(os.path.dirname(model_path), "cnn_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion Matrix 저장됨: {cm_path}")

if __name__ == "__main__":
    evaluate_model()
