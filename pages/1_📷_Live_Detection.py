"""
📷 Live Detection — Real-time webcam gesture recognition.
Uses streamlit-webrtc for live video streaming with hand detection.
"""
import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.hand_detector import HandDetector
from src.gesture_classifier import GestureClassifier
from src.visualizer import draw_prediction_overlay, draw_fps_counter, draw_no_hand_message
from src.utils import FPSCounter, GestureHistory

st.set_page_config(page_title="Live Detection — GestureAI", page_icon="📷", layout="wide")

# ─── CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }

    .live-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .live-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ffaa, #00c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .live-desc {
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    .prediction-box {
        background: linear-gradient(145deg, rgba(25,25,45,0.95), rgba(15,15,35,0.98));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .pred-gesture {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .pred-confidence {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffaa, #00c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .pred-command {
        font-size: 0.85rem;
        color: rgba(0,255,170,0.7);
        margin-top: 0.3rem;
    }

    .history-item {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        border: 1px solid rgba(255,255,255,0.04);
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    .status-active { background: #00ffaa; }
    .status-inactive { background: #ff4444; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────
st.markdown("""
<div class="live-header">
    <div class="live-title">📷 Live Gesture Detection</div>
    <div class="live-desc">
        Real-time hand gesture recognition using your webcam. 
        Show your hand to start detecting gestures instantly.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Initialize Session State ────────────────────────────
if 'gesture_history' not in st.session_state:
    st.session_state.gesture_history = GestureHistory(max_length=config.MAX_HISTORY_LENGTH)
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

# ─── Controls ────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])

with col_ctrl1:
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=0.95, value=config.CONFIDENCE_THRESHOLD,
        step=0.05, help="Minimum confidence to display a prediction"
    )

with col_ctrl2:
    detection_confidence = st.slider(
        "Hand Detection Sensitivity",
        min_value=0.3, max_value=0.95, value=config.MP_DETECTION_CONFIDENCE,
        step=0.05, help="MediaPipe hand detection confidence"
    )

with col_ctrl3:
    show_landmarks = st.checkbox("Show Landmarks", value=True)

# ─── Main Content ────────────────────────────────────────
col_video, col_info = st.columns([3, 1])

with col_video:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border-radius: 12px; padding: 1rem; 
         border: 1px solid rgba(255,255,255,0.06); text-align: center; margin-bottom: 1rem;">
        <span class="status-dot status-active"></span>
        <span style="color: rgba(255,255,255,0.6); font-size: 0.85rem;">
            Camera feed — click Start to begin detection
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Camera capture using Streamlit's native camera
    run_camera = st.checkbox("🎥 Start Camera", value=False, key="camera_toggle")
    FRAME_WINDOW = st.image([])

with col_info:
    # Prediction display
    pred_placeholder = st.empty()
    st.markdown("---")
    st.markdown("**📜 Gesture History**")
    history_placeholder = st.empty()

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.gesture_history.clear()

# ─── Camera Loop ─────────────────────────────────────────
if run_camera:
    detector = HandDetector(
        detection_confidence=detection_confidence,
        static_mode=False,
    )
    classifier = GestureClassifier()
    fps_counter = FPSCounter()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please check your camera permissions.")
    else:
        last_log_time = 0

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read from camera.")
                break

            fps_counter.tick()
            frame = cv2.flip(frame, 1)  # Mirror

            # Detect hands
            results, rgb_frame = detector.detect(frame)

            if show_landmarks:
                frame = detector.draw_landmarks(frame, results)

            hands_data = detector.get_landmarks(results, frame.shape)

            current_prediction = None

            if hands_data and classifier.is_ready():
                for i, hand in enumerate(hands_data):
                    roi = detector.extract_hand_roi(frame, hand['bbox'])
                    if roi is not None:
                        prediction = classifier.predict(roi)
                        if prediction['confidence'] >= confidence_threshold:
                            current_prediction = prediction
                            frame = draw_prediction_overlay(frame, prediction, hand['bbox'], i)

                            # Log to history (rate-limited)
                            now = time.time()
                            if now - last_log_time > 1.5:
                                st.session_state.gesture_history.add(prediction)
                                last_log_time = now
            elif not hands_data:
                frame = draw_no_hand_message(frame)

            # Draw FPS
            frame = draw_fps_counter(frame, fps_counter.get_fps())

            # Display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update prediction display
            if current_prediction and current_prediction['confidence'] >= confidence_threshold:
                pred_placeholder.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 2.5rem; text-align: center;">{current_prediction['emoji']}</div>
                    <div class="pred-gesture" style="text-align: center;">{current_prediction['class_name']}</div>
                    <div class="pred-confidence" style="text-align: center;">{current_prediction['confidence']*100:.1f}%</div>
                    <div class="pred-command" style="text-align: center;">→ {current_prediction['command']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                pred_placeholder.markdown("""
                <div class="prediction-box" style="text-align: center;">
                    <div style="font-size: 2rem; opacity: 0.3;">🖐️</div>
                    <div style="color: rgba(255,255,255,0.4); font-size: 0.9rem;">Waiting for gesture...</div>
                </div>
                """, unsafe_allow_html=True)

            # Update history display
            recent = st.session_state.gesture_history.get_recent(8)
            if recent:
                hist_html = ""
                for entry in recent:
                    hist_html += f"""
                    <div class="history-item">
                        <span>{entry['emoji']} {entry['gesture']}</span>
                        <span style="color: rgba(0,255,170,0.6);">{entry['confidence']*100:.0f}% · {entry['timestamp']}</span>
                    </div>
                    """
                history_placeholder.markdown(hist_html, unsafe_allow_html=True)

        cap.release()
        detector.close()
else:
    # Show placeholder when camera is off
    st.info("👆 Toggle **Start Camera** above to begin real-time gesture detection.")

    pred_placeholder.markdown("""
    <div class="prediction-box" style="text-align: center;">
        <div style="font-size: 2rem; opacity: 0.3;">🖐️</div>
        <div style="color: rgba(255,255,255,0.4); font-size: 0.9rem;">Camera is off</div>
    </div>
    """, unsafe_allow_html=True)
