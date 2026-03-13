import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.augment import get_val_transforms
from models.custom_dl import CustomDLModel
from utils.metrics import calculate_metrics, plot_confusion_matrix


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- 모델 로드 ---
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = CustomDLModel(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"모델: {model_name} (epoch {checkpoint['epoch']}, val_acc {checkpoint['val_acc']:.1f}%)")
    print(f"클래스: {class_to_idx}")

    # --- 테스트 데이터 ---
    test_dir = os.path.join(args.data_dir, 'test')
    test_dataset = ImageFolder(test_dir, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    print(f"Test: {len(test_dataset)}장")

    # --- 평가 ---
    all_preds = []
    all_labels = []
    total_time = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start = time.perf_counter()
            outputs = model(images)
            total_time += time.perf_counter() - start

            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels) * 100

    # F1, Precision, Recall (utils/metrics.py 활용)
    metrics = calculate_metrics(all_labels, all_preds)

    # 추론 속도
    avg_time = total_time / len(all_labels)

    print(f"\n{'='*40}")
    print(f"Test Accuracy:  {accuracy:.1f}%")
    print(f"F1-Score:       {metrics['F1']:.4f}")
    print(f"Precision:      {metrics['Precision']:.4f}")
    print(f"Recall:         {metrics['Recall']:.4f}")
    print(f"추론 속도:      {avg_time*1000:.1f} ms/장")
    print(f"{'='*40}")

    # Confusion Matrix
    class_names = [idx_to_class[i] for i in range(num_classes)]
    plot_confusion_matrix(all_labels, all_preds, class_names)


def get_args():
    parser = argparse.ArgumentParser(description="바나나 숙성도 분류 - 평가")
    parser.add_argument('--weights', type=str, required=True, help='.pth 모델 파일 경로')
    parser.add_argument('--data_dir', type=str, default='dataset/processed')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(get_args())
