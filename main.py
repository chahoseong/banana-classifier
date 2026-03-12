import argparse
import os
import time

import torch
from PIL import Image

from data.preprocess import process_and_save_dataset
from data.augment import get_val_transforms
from models.custom_dl import CustomDLModel, SUPPORTED_MODELS


def main():
    parser = argparse.ArgumentParser(description="바나나 숙성도 분류 메인 파이프라인")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'infer', 'preprocess'],
                        help='작업 모드 선택')
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                        choices=SUPPORTED_MODELS, help='모델 선택')
    parser.add_argument('--data_dir', type=str, default='dataset/processed')
    parser.add_argument('--weights', type=str, default=None, help='.pth 모델 파일 경로')
    parser.add_argument('--image', type=str, default=None, help='추론할 이미지 경로')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='weights')
    args = parser.parse_args()

    print(f"[{args.mode.upper()}] 모드로 시작합니다...")

    if args.mode == 'preprocess':
        raw_dir = os.path.join('dataset', 'raw')
        processed_dir = os.path.join('dataset', 'processed')

        print(f"원본 경로: {raw_dir}")
        print(f"저장 경로: {processed_dir}")

        if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
            print(f"[Error] '{raw_dir}' 경로 내에 과일 클래스 폴더(예: unripe, ripe 등)와 원본 이미지를 먼저 넣어주세요.")
            return

        print("데이터 전처리 및 분할 로직을 시작합니다.")
        process_and_save_dataset(raw_dir, processed_dir, target_size=(224, 224), seed=42)

    elif args.mode == 'train':
        from train import train as run_train
        run_train(args)

    elif args.mode == 'eval':
        if not args.weights:
            print("[Error] --weights 인자로 .pth 모델 파일 경로를 지정해 주세요.")
            return
        from evaluate import evaluate as run_eval
        run_eval(args)

    elif args.mode == 'infer':
        if not args.weights or not args.image:
            print("[Error] --weights 와 --image 인자를 모두 지정해 주세요.")
            return
        run_inference(args)


def run_inference(args):
    """단일 이미지 추론: 클래스명 + 신뢰도 출력"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model_name = checkpoint['model_name']
    num_classes = checkpoint['num_classes']
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = CustomDLModel(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 이미지 로드 및 전처리
    transform = get_val_transforms()
    image = Image.open(args.image).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # 추론
    start = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
    elapsed = time.perf_counter() - start

    confidence, pred_idx = probs.max(1)
    pred_class = idx_to_class[pred_idx.item()]

    print(f"\n{'='*40}")
    print(f"이미지:    {args.image}")
    print(f"예측 결과: {pred_class}")
    print(f"신뢰도:    {confidence.item()*100:.1f}%")
    print(f"추론 시간: {elapsed*1000:.1f} ms")
    print(f"{'='*40}")

    # 전체 클래스별 확률 출력
    print("\n[클래스별 확률]")
    for i in range(num_classes):
        print(f"  {idx_to_class[i]:>10s}: {probs[0][i].item()*100:.1f}%")


if __name__ == "__main__":
    main()
