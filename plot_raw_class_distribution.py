import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_raw_class_distribution(raw_dir='dataset/raw'):
    print(f"'{raw_dir}' 폴더 내의 원본 이미지 개수를 확인합니다...")
    
    # 1. 4개의 클래스(폴더) 확인 및 이미지 개수 카운트
    classes = ['unripe', 'ripe', 'overripe', 'dispose']
    counts = []
    
    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        if os.path.exists(cls_path):
            # 숨김 파일 등을 제외한 실제 이미지 파일 개수 카운트
            images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            counts.append(len(images))
        else:
            print(f"⚠️ 경고: '{cls_path}' 폴더가 존재하지 않아서 개수를 0으로 처리합니다.")
            counts.append(0)
            
    # 한글 폰트 설정 (깨짐 방지)
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass # 폰트가 없으면 기본값 사용

    # 2. 바 차트 그리기
    plt.figure(figsize=(10, 6))
    
    # Seaborn 막대 그래프
    ax = sns.barplot(x=classes, y=counts, palette='viridis')
    
    plt.title('바나나 원본 데이터셋(`dataset/raw`) 클래스별 이미지 개수', fontsize=16, pad=15)
    plt.xlabel('숙성도 클래스 (Class)', fontsize=12)
    plt.ylabel('이미지 장수 (Count)', fontsize=12)
    
    # 3. 막대 그래프 위에 정확한 숫자 표시
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.annotate(f'{int(height)}장', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')

    # Y축 최대값 약간 여유있게 올리기 (글자 짤림 방지)
    max_count = max(counts) if counts else 100
    plt.ylim(0, max_count * 1.15)
    
    plt.tight_layout()
    
    # 4. 파일로 저장
    output_filename = 'class_distribution.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\n✅ 원본 데이터 개수 막대 그래프를 '{output_filename}' 파일로 저장했습니다.")
    
    # 요약 출력
    print("-" * 30)
    for c, n in zip(classes, counts):
        print(f" - {c.capitalize()}: {n}장")
    print(f" = 총계: {sum(counts)}장")
    print("-" * 30)

if __name__ == '__main__':
    sns.set_theme(style="whitegrid") # 깔끔한 배경
    plot_raw_class_distribution()
