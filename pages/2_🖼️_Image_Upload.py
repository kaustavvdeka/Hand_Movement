"""
🖼️ Image Upload — Gesture detection from uploaded images.
"""
import streamlit as st
import cv2
import numpy as np
import os, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.hand_detector import HandDetector
from src.gesture_classifier import GestureClassifier
from src.visualizer import draw_prediction_overlay, create_landmark_visualization
from src.utils import preprocess_uploaded_image

st.set_page_config(page_title="Image Upload — GestureAI", page_icon="🖼️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
.stApp{font-family:'Inter',sans-serif}
#MainMenu,footer,header{visibility:hidden}
.page-header{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);border-radius:16px;padding:2rem;margin-bottom:1.5rem;border:1px solid rgba(255,255,255,.06)}
.page-title{font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#00ffaa,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.page-desc{color:rgba(255,255,255,.6);font-size:.9rem;margin-top:.3rem}
.result-card{background:linear-gradient(145deg,rgba(25,25,45,.95),rgba(15,15,35,.98));border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:1.5rem;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">🖼️ Image Analysis</div>
    <div class="page-desc">Upload an image to detect and classify hand gestures. Supports PNG, JPG, JPEG, BMP formats.</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a hand gesture image", type=["png","jpg","jpeg","bmp"], key="image_upload")

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img_bgr, preprocessed = preprocess_uploaded_image(image_bytes)
    if img_bgr is None:
        st.error("❌ Could not decode the uploaded image.")
    else:
        detector = HandDetector(static_mode=True, detection_confidence=0.5)
        classifier = GestureClassifier()
        col_img, col_result = st.columns([3, 2])

        with col_img:
            results, _ = detector.detect(img_bgr)
            hands_data = detector.get_landmarks(results, img_bgr.shape)
            annotated = detector.draw_landmarks(img_bgr, results)
            predictions = []
            if hands_data and classifier.is_ready():
                for i, hand in enumerate(hands_data):
                    roi = detector.extract_hand_roi(img_bgr, hand['bbox'])
                    if roi is not None:
                        pred = classifier.predict(roi)
                        predictions.append((pred, hand))
                        annotated = draw_prediction_overlay(annotated, pred, hand['bbox'], i)
            if not predictions and classifier.is_ready():
                pred = classifier.predict(preprocessed)
                predictions.append((pred, None))
            st.markdown("**📸 Analyzed Image**")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            if hands_data:
                st.markdown("**🔍 Hand Landmarks**")
                for i, hand in enumerate(hands_data):
                    viz = create_landmark_visualization(hand['landmarks'])
                    st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB), caption=f"Hand {i+1} — {hand['handedness']}", width=300)

        with col_result:
            if predictions:
                for pred, hand in predictions:
                    hl = f" ({hand['handedness']})" if hand else ""
                    st.markdown(f"""<div class="result-card" style="text-align:center">
                        <div style="font-size:3rem">{pred['emoji']}</div>
                        <div style="font-size:1.5rem;font-weight:700;color:#fff">{pred['class_name']}{hl}</div>
                        <div style="font-size:2.5rem;font-weight:800;background:linear-gradient(135deg,#00ffaa,#00c8ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{pred['confidence']*100:.1f}%</div>
                        <div style="color:rgba(0,255,170,.7);font-size:.9rem">Command: {pred['command']}</div>
                    </div>""", unsafe_allow_html=True)
                    if pred['all_probabilities']:
                        st.markdown("**📊 Class Probabilities**")
                        for cls, prob in sorted(pred['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                            emoji = config.GESTURE_EMOJIS.get(cls, "")
                            col = "#00ffaa" if prob == max(pred['all_probabilities'].values()) else "#3b4a6b"
                            st.progress(prob, text=f"{emoji} {cls}: {prob*100:.1f}%")
            else:
                st.info("No trained model available. Run `python train_model.py` first.")
        detector.close()
else:
    st.markdown("""<div style="text-align:center;padding:4rem 2rem;background:rgba(255,255,255,.02);border-radius:16px;border:2px dashed rgba(255,255,255,.1)">
        <div style="font-size:4rem;opacity:.3">🖼️</div>
        <div style="color:rgba(255,255,255,.5);font-size:1.1rem;margin-top:1rem">Drag and drop an image here, or click to browse</div>
    </div>""", unsafe_allow_html=True)
