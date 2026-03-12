import argparse
# from data.preprocess import preprocess_image
# from data.augment import get_train_transforms, get_val_transforms
# from models.custom_dl import CustomDLModel

def main():
    parser = argparse.ArgumentParser(description="바나나 숙성도 분류 메인 파이프라인")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'infer'], help='작업 모드 선택')
    args = parser.parse_args()

    print(f"[{args.mode.upper()}] 모드로 시작합니다...")
    
    if args.mode == 'train':
        print("학습 로직 구현")
    elif args.mode == 'eval':
        print("평가 로직 구현")
    elif args.mode == 'infer':
        print("추론 로직 구현")

if __name__ == "__main__":
    main()
