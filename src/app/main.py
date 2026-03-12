import streamlit as st
import numpy as np
import cv2
import sys
import os
from PIL import Image

# 프로젝트 루트 디렉토리를 path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.inference.inference_engine import InferenceEngine, DummyModel

# 1. 페이지 설정
st.set_page_config(
    page_title="Banana Ripe Classifier",
    page_icon="🍌",
    layout="wide"
)

# 2. 시스템 초기화 (세션 상태 관리)
if 'engine' not in st.session_state:
    st.session_state.engine = InferenceEngine()
    # 초기 모델로 Dummy 모델 설정
    st.session_state.engine.set_model(DummyModel())

# 3. 사이드바 - 설정
st.sidebar.title("⚙️ 설정")
model_type = st.sidebar.selectbox(
    "사용할 모델 선택",
    ["Dummy ML Model", "Deep Learning Model (준비 중)"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Banana Ripe Classifier**
    
    웹캠에 바나나를 보여주세요.
    AI가 숙성도를 4단계로 분류합니다.
    """
)

# 4. 메인 화면 - 타이틀 및 레이아웃
st.title("🍌 바나나 숙성도 자동 판별 시스템")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 실시간 웹캠")
    # Streamlit 내장 카메라 입력 사용 (개발 속도 및 범용성)
    img_file_buffer = st.camera_input("바나나를 카메라 앞에 비춰주세요")

with col2:
    st.subheader("📉 판별 결과")
    
    if img_file_buffer is not None:
        # 이미지를 numpy array로 변환
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        
        # 추론 수행
        with st.spinner('판별 중...'):
            category, confidence = st.session_state.engine.infer(img_array)
        
        # 결과 표시
        color_map = {
            'unripe': '#4CAF50',    # Green
            'ripe': '#FFEB3B',      # Yellow
            'overripe': '#FF9800',  # Orange
            'dispose': '#F44336'    # Red
        }
        
        st.markdown(f"### 예측 결과: <span style='color:{color_map.get(category, '#000')}'>{category.upper()}</span>", unsafe_allow_html=True)
        st.write(f"**신뢰도:** {confidence:.2%}")
        st.progress(confidence)
        
        # 숙성도별 가이드 메시지
        messages = {
            'unripe': "아직 덜 익었어요. 며칠 더 기다려주세요! ⏳",
            'ripe': "지금이 가장 맛있을 때입니다! 맛있게 드세요! 😋",
            'overripe': "빨리 드시는 게 좋아요. 스무디나 빵을 만드는 데 추천합니다! 🍞",
            'dispose': "아쉽지만 먹기에 적절하지 않습니다. 폐기를 권장합니다. 🚮"
        }
        st.success(messages.get(category, ""))
        
    else:
        st.warning("카메라 입력을 기다리고 있습니다...")

# 5. 하단 정보
st.markdown("---")
st.caption("© 2026 Banana Ripe Classifier Team | Powered by Streamlit & AI")
