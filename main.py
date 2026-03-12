import argparse
import os

from data.preprocess import process_and_save_dataset
# from data.augment import get_train_transforms, get_val_transforms
# from models.custom_dl import CustomDLModel

def main():
    parser = argparse.ArgumentParser(description="바나나 숙성도 분류 메인 파이프라인")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'infer', 'preprocess'], help='작업 모드 선택')
    args = parser.parse_args()

    print(f"[{args.mode.upper()}] 모드로 시작합니다...")
    
    if args.mode == 'preprocess':
        # 로직 연동
        raw_dir = os.path.join('dataset', 'raw')
        processed_dir = os.path.join('dataset', 'processed')
        
        print(f"원본 경로: {raw_dir}")
        print(f"저장 경로: {processed_dir}")
        
        # 원본 데이터 폴더가 비어있는지 체크
        if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
            print(f"[Error] '{raw_dir}' 경로 내에 과일 클래스 폴더(예: unripe, ripe 등)와 원본 이미지를 먼저 넣어주세요.")
            return

        print("데이터 전처리 및 분할 로직을 시작합니다.")
        process_and_save_dataset(raw_dir, processed_dir, target_size=(224, 224), seed=42)

    elif args.mode == 'train':
        print("학습 로직 구현")
    elif args.mode == 'eval':
        print("평가 로직 구현")
    elif args.mode == 'infer':
        print("추론 로직 구현")

if __name__ == "__main__":
    main()
