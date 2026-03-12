import torch
from torchvision import transforms

def get_train_transforms(image_size=224):
    """
    학습 데이터셋을 위한 확률적 데이터 증강(Augmentation) 파이프라인.
    [팀원 2 담당 영역]
    
    Returns:
        torchvision.transforms.Compose
    """
    # TODO: RandomFlip, ColorJitter, Normalize 등 적용
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(image_size=224):
    """
    검증 및 테스트 데이터셋을 위한 정적 변환(Transform) 파이프라인.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
