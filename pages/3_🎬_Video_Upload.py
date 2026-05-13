"""
🎬 Video Upload — Process uploaded video files for gesture recognition.
"""
import streamlit as st
import cv2
import numpy as np
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.hand_detector import HandDetector
from src.gesture_classifier import GestureClassifier
from src.visualizer import draw_prediction_overlay, draw_no_hand_message
from src.utils import GestureHistory

st.set_page_config(page_title="Video Upload — GestureAI", page_icon="🎬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
.stApp{font-family:'Inter',sans-serif}
#MainMenu,footer,header{visibility:hidden}
.page-header{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);border-radius:16px;padding:2rem;margin-bottom:1.5rem;border:1px solid rgba(255,255,255,.06)}
.page-title{font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#00ffaa,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.page-desc{color:rgba(255,255,255,.6);font-size:.9rem;margin-top:.3rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">🎬 Video Analysis</div>
    <div class="page-desc">Upload a video file to perform frame-by-frame gesture recognition with timeline visualization.</div>
</div>
""", unsafe_allow_html=True)

uploaded_video = st.file_uploader("Upload a video file", type=["mp4","avi","mov","mkv"], key="video_upload")

if uploaded_video is not None:
    # Save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Total Frames", f"{total_frames}")
    col_info2.metric("FPS", f"{fps:.1f}")
    col_info3.metric("Duration", f"{duration:.1f}s")

    # Processing controls
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        skip_frames = st.slider("Process every N frames", 1, 10, 3, help="Skip frames to speed up processing")
    with col_c2:
        conf_thresh = st.slider("Confidence Threshold", 0.1, 0.95, 0.45, 0.05)

    if st.button("🚀 Process Video", type="primary", use_container_width=True):
        detector = HandDetector(static_mode=True, detection_confidence=0.5)
        classifier = GestureClassifier()
        history = GestureHistory(max_length=500)

        progress_bar = st.progress(0, text="Processing...")
        frame_display = st.empty()
        timeline_data = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                results, _ = detector.detect(frame)
                hands_data = detector.get_landmarks(results, frame.shape)
                annotated = detector.draw_landmarks(frame, results)

                if hands_data and classifier.is_ready():
                    for i, hand in enumerate(hands_data):
                        roi = detector.extract_hand_roi(frame, hand['bbox'])
                        if roi is not None:
                            pred = classifier.predict(roi)
                            if pred['confidence'] >= conf_thresh:
                                annotated = draw_prediction_overlay(annotated, pred, hand['bbox'], i)
                                timeline_data.append({
                                    'frame': frame_idx,
                                    'time': frame_idx / fps if fps > 0 else 0,
                                    'gesture': pred['class_name'],
                                    'confidence': pred['confidence'],
                                    'emoji': pred['emoji']
                                })
                                history.add(pred)

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_display.image(frame_rgb, use_container_width=True)

            frame_idx += 1
            progress_bar.progress(min(frame_idx / total_frames, 1.0), text=f"Frame {frame_idx}/{total_frames}")

        cap.release()
        detector.close()
        progress_bar.progress(1.0, text="✅ Processing Complete!")

        # Results summary
        st.markdown("---")
        st.markdown("### 📊 Detection Timeline")

        if timeline_data:
            import pandas as pd
            import plotly.express as px

            df = pd.DataFrame(timeline_data)
            fig = px.scatter(df, x='time', y='gesture', color='gesture', size='confidence',
                           title='Gesture Detection Timeline', labels={'time': 'Time (s)', 'gesture': 'Gesture'},
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(
                plot_bgcolor='rgba(15,15,35,0.95)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='rgba(255,255,255,0.7)', height=400,
                title_font_size=16
            )
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            st.markdown("### 📋 Detection Summary")
            stats = history.get_stats()
            stat_cols = st.columns(min(len(stats), 5))
            for i, (gesture, count) in enumerate(sorted(stats.items(), key=lambda x: x[1], reverse=True)):
                with stat_cols[i % len(stat_cols)]:
                    emoji = config.GESTURE_EMOJIS.get(gesture, "🖐️")
                    st.metric(f"{emoji} {gesture}", f"{count} detections")
        else:
            st.info("No gestures detected in the video. Try lowering the confidence threshold.")

    # Cleanup
    os.unlink(tfile.name)
else:
    st.markdown("""<div style="text-align:center;padding:4rem 2rem;background:rgba(255,255,255,.02);border-radius:16px;border:2px dashed rgba(255,255,255,.1)">
        <div style="font-size:4rem;opacity:.3">🎬</div>
        <div style="color:rgba(255,255,255,.5);font-size:1.1rem;margin-top:1rem">Upload a video to analyze gestures frame by frame</div>
    </div>""", unsafe_allow_html=True)
