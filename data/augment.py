from torchvision import transforms


def get_train_transforms(image_size=224):
    """
    학습 데이터셋을 위한 확률적 데이터 증강(Augmentation) 파이프라인.
    [팀원 2 담당 영역]

    이미지가 이미 224x224로 전처리되어 있으므로,
    RandomResizedCrop으로 약간의 스케일 변화를 주고
    Flip, Rotation, ColorJitter로 다양성을 확보합니다.

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size=224):
    """
    검증 및 테스트 데이터셋을 위한 정적 변환(Transform) 파이프라인.
    증강 없이 정규화만 적용합니다.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
