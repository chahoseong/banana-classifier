import os
import time
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.augment import get_train_transforms, get_val_transforms
from models.custom_dl import CustomDLModel


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- 데이터 로딩 ---
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    train_dataset = ImageFolder(train_dir, transform=get_train_transforms())
    val_dataset = ImageFolder(val_dir, transform=get_val_transforms())

    print(f"클래스 매핑: {train_dataset.class_to_idx}")
    print(f"Train: {len(train_dataset)}장, Val: {len(val_dataset)}장")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # --- 모델 ---
    model = CustomDLModel(model_name=args.model, num_classes=len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=3)

    # --- 학습 루프 ---
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = None

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total * 100

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total * 100

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.1e}")

        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # 네이밍 규칙: [모델명]_ep[에폭]_acc[검증정확도]_[MMDD].pth
            date_str = datetime.datetime.now().strftime('%m%d')
            filename = f"{args.model}_ep{epoch}_acc{int(val_acc)}_{date_str}.pth"
            save_dir = args.save_dir
            os.makedirs(save_dir, exist_ok=True)
            best_model_path = os.path.join(save_dir, filename)

            torch.save({
                'model_name': args.model,
                'num_classes': len(train_dataset.classes),
                'class_to_idx': train_dataset.class_to_idx,
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"  → Best model saved: {filename}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= args.patience:
            print(f"\nEarly Stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\n학습 완료. Best Val Accuracy: {best_val_acc:.1f}%")
    if best_model_path:
        print(f"Best model: {best_model_path}")


def get_args():
    parser = argparse.ArgumentParser(description="바나나 숙성도 분류 - 학습")
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'resnet18', 'resnet50'])
    parser.add_argument('--data_dir', type=str, default='dataset/processed')
    parser.add_argument('--save_dir', type=str, default='weights')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    train(get_args())
