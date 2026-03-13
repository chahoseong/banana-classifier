import streamlit as st
import numpy as np
import cv2
import sys
import os
from PIL import Image

# 프로젝트 루트 디렉토리를 path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.inference.inference_engine import InferenceEngine, DummyModel
from src.models.ml_baseline import MLBaselineModel
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import threading

# 1. 페이지 설정
st.set_page_config(
    page_title="Banana Ripe Classifier",
    page_icon="🍌",
    layout="wide"
)

# 2. 시스템 초기화 (세션 상태 관리)
if 'engine' not in st.session_state:
    st.session_state.engine = InferenceEngine()
    st.session_state.engine.set_model(DummyModel())

# 3. 비디오 처리 클래스 정의
class BananaVideoProcessor(VideoProcessorBase):
    def __init__(self, engine):
        self.engine = engine
        self.result_lock = threading.Lock()
        self.category = None
        self.confidence = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        
        # 추론 수행
        try:
            category, confidence = self.engine.infer(img)
            with self.result_lock:
                self.category = category
                self.confidence = confidence
        except Exception as e:
            print(f"Inference error: {e}")
            
        return frame

# 4. 사이드바 - 설정
st.sidebar.title("⚙️ 설정")
model_option = st.sidebar.selectbox(
    "사용할 모델 선택",
    ["Dummy ML Model", "kNN Baseline Model", "Deep Learning Model (준비 중)"]
)

# 모델 변경 시 적용
if model_option == "kNN Baseline Model":
    if not isinstance(st.session_state.engine.model, MLBaselineModel):
        try:
            st.session_state.engine.set_model(MLBaselineModel(), "models/saved")
            st.sidebar.success("kNN 모델 로드 완료!")
        except Exception as e:
            st.sidebar.error(f"모델 로드 실패: {e}")
elif model_option == "Dummy ML Model":
    if not isinstance(st.session_state.engine.model, DummyModel):
        st.session_state.engine.set_model(DummyModel())

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Banana Ripe Classifier**
    
    웹캠에 바나나를 보여주세요.
    AI가 숙성도를 실시간으로 분류합니다.
    """
)

# 5. 메인 화면 - 타이틀 및 레이아웃
st.title("🍌 바나나 숙성도 실시간 판별 시스템")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 실시간 웹캠 스트림")
    # 세션 상태를 직접 접근하지 않고 로컬 변수로 캡처하여 전달 (작업자 스레드 오류 방지)
    current_engine = st.session_state.engine
    ctx = webrtc_streamer(
        key="banana-classifier",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: BananaVideoProcessor(current_engine),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📉 실시간 판별 결과")
    
    result_container = st.empty()
    
    # 실시간 결과를 계속 업데이트하기 위한 루프 (Streamlit 특성상 상태 변화 시 리런되지만 WebRTC는 별도)
    if ctx.video_processor:
        with ctx.video_processor.result_lock:
            category = ctx.video_processor.category
            confidence = ctx.video_processor.confidence
        
        if category:
            color_map = {
                'unripe': '#4CAF50',    # Green
                'ripe': '#FFEB3B',      # Yellow
                'overripe': '#FF9800',  # Orange
                'dispose': '#F44336'    # Red
            }
            
            with result_container.container():
                st.markdown(f"### 예측 결과: <span style='color:{color_map.get(category, '#000')}'>{category.upper()}</span>", unsafe_allow_html=True)
                st.write(f"**신뢰도:** {confidence:.2%}")
                st.progress(confidence)
                
                messages = {
                    'unripe': "아직 덜 익었어요. 며칠 더 기다려주세요! ⏳",
                    'ripe': "지금이 가장 맛있을 때입니다! 맛있게 드세요! 😋",
                    'overripe': "빨리 드시는 게 좋아요. 스무디나 빵을 만드는 데 추천합니다! 🍞",
                    'dispose': "아쉽지만 먹기에 적절하지 않습니다. 폐기를 권장합니다. 🚮"
                }
                st.success(messages.get(category, ""))
        else:
            result_container.info("카메라 분석을 시작하려면 Start 버튼을 눌러주세요.")
    else:
        st.warning("웹캠 연결을 대기 중입니다...")

# 실시간 업데이트를 위한 Polling 로직
if ctx.state.playing:
    import time
    time.sleep(0.5)  # 0.5초마다 UI 갱신
    st.rerun()

# 6. 하단 정보
st.markdown("---")
st.caption("© 2026 Banana Ripe Classifier Team | Powered by Streamlit & WebRTC")
