import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
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
        
        # 이미지 로드 (BGR)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            # 실패 시 빈 이미지 반환 (실제로는 필터링 권장)
            image_bgr = np.zeros((300, 300, 3), dtype=np.uint8)
        
        image_bgr = cv2.resize(image_bgr, (224, 224)) # EfficientNet 기본 입력 사이즈
        
        if self.use_mask:
            # 강력한 마스킹 적용
            mask = image_processor.create_banana_mask(image_bgr)
            # 마스크를 3채널로 확장하여 이미지에 적용 (배경을 검은색으로)
            mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
            image_masked = (image_bgr * mask_3ch).astype(np.uint8)
            image_rgb = cv2.cvtColor(image_masked, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            image_rgb = self.transform(image_rgb)
            
        return image_rgb, self.class_to_idx[label]

def get_data_loaders(dataset_path, batch_size=16):
    file_paths = []
    labels = []
    
    class_mapping = {
        "Unripe": "unripe",
        "Ripe": "ripe",
        "Overripe": "overripe",
        "Dispose": "dispose"
    }
    
    for folder_name, label in class_mapping.items():
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_paths.append(os.path.join(folder_path, file_name))
                labels.append(label)
                
    # 학습/검증 분할
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 데이터 증강 (Augmentation)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = BananaDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = BananaDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model():
    # 설정
    dataset_path = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\dataset\raw"
    artifacts_dir = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\ml\artifacts"
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"사용 가능 디바이스: {device}")
    
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)
        
    train_loader, val_loader = get_data_loaders(dataset_path, batch_size)
    
    # 모델 정의: EfficientNet-B0 (Pre-trained)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # 마지막 레이어 교체 (4개 클래스)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 4)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # 학습 단계
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        # 최고 성능 모델 저장
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            save_path = os.path.join(artifacts_dir, "banana_cnn_v1.pth")
            torch.save(model.state_dict(), save_path)
            print(f"새로운 최고 모델 저장됨: {save_path} (Acc: {best_acc:.4f})")
            
    print(f"\n학습 완료. 최적 Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train_model()
