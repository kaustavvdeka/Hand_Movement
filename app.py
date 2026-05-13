"""
🖐️ Hand Gesture Recognition System
Main Streamlit Application — Home Dashboard

A premium AI-powered gesture recognition platform with real-time detection,
visualization, and analytics.
"""
import streamlit as st
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─── Page Configuration ─────────────────────────────────
st.set_page_config(
    page_title="GestureAI — Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0, 255, 170, 0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffaa 0%, #00c8ff 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 300;
        line-height: 1.6;
        max-width: 700px;
        position: relative;
        z-index: 1;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(145deg, rgba(30, 30, 50, 0.9), rgba(20, 20, 40, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.8rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 255, 170, 0.3);
        box-shadow: 0 12px 40px rgba(0, 255, 170, 0.08);
    }
    .feature-icon {
        font-size: 2.2rem;
        margin-bottom: 0.8rem;
        display: inline-block;
    }
    .feature-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 0.88rem;
        color: rgba(255, 255, 255, 0.55);
        line-height: 1.5;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(145deg, rgba(25, 25, 45, 0.95), rgba(15, 15, 35, 0.98));
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00ffaa, #00c8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* Gesture grid */
    .gesture-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        margin: 0.3rem;
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        transition: all 0.2s;
    }
    .gesture-chip:hover {
        background: rgba(0, 255, 170, 0.08);
        border-color: rgba(0, 255, 170, 0.3);
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 255, 170, 0.2);
    }

    /* App cards */
    .app-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s;
    }
    .app-card:hover {
        background: rgba(0, 255, 170, 0.05);
        border-color: rgba(0, 255, 170, 0.2);
    }
    .app-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .app-name {
        font-size: 0.9rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.8);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: rgba(255, 255, 255, 0.7);
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem;">🖐️</div>
        <div style="font-size: 1.3rem; font-weight: 700; 
            background: linear-gradient(135deg, #00ffaa, #00c8ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            GestureAI
        </div>
        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.4); margin-top: 0.2rem;">
            v2.0 — Powered by Deep Learning
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model status
    model_exists = os.path.exists(config.MODEL_PATH)
    if model_exists:
        st.success("✅ Model Loaded", icon="🧠")
        # Load training history for stats
        if os.path.exists(config.HISTORY_PATH):
            hist = np.load(config.HISTORY_PATH, allow_pickle=True).item()
            acc = hist.get('test_accuracy', 0)
            st.metric("Test Accuracy", f"{acc * 100:.1f}%")
    else:
        st.warning("⚠️ No trained model found", icon="🔧")
        st.caption("Run `python train_model.py` to train")

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: rgba(255,255,255,0.4); text-align: center;">
        Built with MediaPipe + TensorFlow<br>
        © 2026 GestureAI
    </div>
    """, unsafe_allow_html=True)

# ─── Hero Section ────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">GestureAI</div>
    <div class="hero-subtitle">
        Advanced AI-powered Hand Gesture Recognition System. Detect, track, and classify
        hand gestures in real-time using Computer Vision and Deep Learning for seamless
        human-computer interaction.
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Stats Row ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">10</div>
        <div class="stat-label">Gesture Classes</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">20K+</div>
        <div class="stat-label">Training Images</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">30+</div>
        <div class="stat-label">FPS Real-time</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    acc_display = "—"
    if os.path.exists(config.HISTORY_PATH):
        try:
            hist = np.load(config.HISTORY_PATH, allow_pickle=True).item()
            acc_display = f"{hist.get('test_accuracy', 0) * 100:.1f}%"
        except:
            pass
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{acc_display}</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Features Section ───────────────────────────────────
st.markdown('<div class="section-header">✨ Key Features</div>', unsafe_allow_html=True)

feat_cols = st.columns(3)
features = [
    ("📷", "Live Detection", "Real-time gesture recognition from your webcam with instant prediction display and confidence scoring."),
    ("🖼️", "Image Analysis", "Upload images for gesture detection. Supports multiple formats with automatic hand landmark extraction."),
    ("🎬", "Video Processing", "Process video files for gesture recognition. Frame-by-frame analysis with timeline visualization."),
    ("📊", "Analytics Dashboard", "Comprehensive model metrics including confusion matrix, accuracy curves, and per-class performance."),
    ("🎯", "High Accuracy", "Deep CNN architecture with data augmentation achieves robust classification across 10 gesture classes."),
    ("⚡", "Real-time Speed", "Optimized inference pipeline delivers 30+ FPS performance for seamless interaction."),
]

for i, (icon, title, desc) in enumerate(features):
    with feat_cols[i % 3]:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

# ─── Supported Gestures ─────────────────────────────────
st.markdown('<div class="section-header">🤟 Supported Gestures</div>', unsafe_allow_html=True)

gesture_html = ""
for class_name, emoji in config.GESTURE_EMOJIS.items():
    command = config.GESTURE_COMMANDS.get(class_name, "")
    gesture_html += f'<div class="gesture-chip">{emoji} {class_name} — <span style="color: rgba(0,255,170,0.7)">{command}</span></div>'

st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 0.3rem;">{gesture_html}</div>', unsafe_allow_html=True)

# ─── Applications Section ───────────────────────────────
st.markdown('<div class="section-header">🌍 Applications</div>', unsafe_allow_html=True)

app_cols = st.columns(6)
applications = [
    ("🏠", "Smart Home"),
    ("🎮", "Gaming"),
    ("📽️", "Presentations"),
    ("♿", "Accessibility"),
    ("🤖", "IoT Control"),
    ("🔬", "HCI Research"),
]

for i, (icon, name) in enumerate(applications):
    with app_cols[i]:
        st.markdown(f"""
        <div class="app-card">
            <div class="app-icon">{icon}</div>
            <div class="app-name">{name}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Quick Start Guide ──────────────────────────────────
st.markdown('<div class="section-header">🚀 Quick Start</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    #### Setup & Training
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Train the CNN model
    python train_model.py

    # 3. Evaluate the model
    python evaluate_model.py

    # 4. Launch the app
    streamlit run app.py
    ```
    """)

with col_b:
    st.markdown("""
    #### Navigation Guide
    Use the **sidebar** to navigate between pages:
    - **📷 Live Detection** — Real-time webcam gesture recognition
    - **🖼️ Image Upload** — Analyze uploaded images
    - **🎬 Video Upload** — Process video files
    - **📊 Analytics** — View model performance metrics
    - **ℹ️ About** — System architecture & documentation
    """)
