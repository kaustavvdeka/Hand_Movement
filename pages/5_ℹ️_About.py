"""
ℹ️ About — System architecture, documentation, and technical details.
"""
import streamlit as st
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title="About — GestureAI", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
.stApp{font-family:'Inter',sans-serif}
#MainMenu,footer,header{visibility:hidden}
.page-header{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);border-radius:16px;padding:2rem;margin-bottom:1.5rem;border:1px solid rgba(255,255,255,.06)}
.page-title{font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#00ffaa,#00c8ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.tech-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:1.2rem;text-align:center;margin-bottom:.5rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">ℹ️ About GestureAI</div>
    <div style="color:rgba(255,255,255,.6);font-size:.9rem;margin-top:.3rem">
        System architecture, technology stack, and documentation.
    </div>
</div>
""", unsafe_allow_html=True)

# System Overview
st.markdown("### 🏗️ System Architecture")
st.markdown("""
```mermaid
graph LR
    A[Input Source] --> B[MediaPipe Hands]
    B --> C[Hand Detector]
    C --> D[ROI Extraction]
    D --> E[CNN Classifier]
    E --> F[Prediction Engine]
    F --> G[Streamlit UI]
    
    style A fill:#1a1a3e,stroke:#00ffaa
    style E fill:#1a1a3e,stroke:#00c8ff
    style G fill:#1a1a3e,stroke:#7c3aed
```
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔧 Technology Stack")
    techs = [
        ("🐍 Python 3.10+", "Core programming language"),
        ("📷 OpenCV", "Computer vision & image processing"),
        ("🤚 MediaPipe", "Real-time hand detection & tracking"),
        ("🧠 TensorFlow/Keras", "Deep learning CNN model"),
        ("📊 Scikit-learn", "Evaluation metrics & preprocessing"),
        ("🎨 Streamlit", "Interactive web dashboard"),
        ("📈 Plotly", "Interactive charts & visualizations"),
    ]
    for name, desc in techs:
        st.markdown(f"""<div class="tech-card">
            <div style="font-weight:600;color:#fff">{name}</div>
            <div style="font-size:.8rem;color:rgba(255,255,255,.5)">{desc}</div>
        </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("### 📁 Project Structure")
    st.code("""
Skillcraft/
├── app.py                    # Main dashboard
├── config.py                 # Configuration
├── train_model.py            # Training script
├── evaluate_model.py         # Evaluation script
├── requirements.txt          # Dependencies
├── src/
│   ├── hand_detector.py      # MediaPipe detection
│   ├── feature_extractor.py  # Landmark features
│   ├── gesture_classifier.py # Model inference
│   ├── model_architecture.py # CNN definition
│   ├── data_pipeline.py      # Data loading
│   ├── visualizer.py         # Drawing utilities
│   └── utils.py              # Helper functions
├── pages/
│   ├── 1_📷_Live_Detection.py
│   ├── 2_🖼️_Image_Upload.py
│   ├── 3_🎬_Video_Upload.py
│   ├── 4_📊_Analytics.py
│   └── 5_ℹ️_About.py
├── models/                   # Trained weights
└── archive/leapGestRecog/    # Dataset
    """, language="text")

st.markdown("### 🤟 Supported Gestures")
gesture_cols = st.columns(5)
for i, (class_name, emoji) in enumerate(config.GESTURE_EMOJIS.items()):
    with gesture_cols[i % 5]:
        cmd = config.GESTURE_COMMANDS.get(class_name, "")
        st.markdown(f"""<div class="tech-card">
            <div style="font-size:2rem">{emoji}</div>
            <div style="font-weight:600;color:#fff;font-size:.9rem">{class_name}</div>
            <div style="font-size:.75rem;color:rgba(0,255,170,.7)">{cmd}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("### 🧠 Model Details")
st.markdown("""
| Parameter | Value |
|-----------|-------|
| Architecture | Custom CNN (4 conv blocks) |
| Input Size | 64 × 64 × 1 (Grayscale) |
| Parameters | ~800K |
| Optimizer | Adam (lr=0.001) |
| Loss | Sparse Categorical Crossentropy |
| Regularization | L2 + Dropout + BatchNorm |
| Augmentation | Rotation, Zoom, Translation, Contrast |
| Training Data | 20,000 images (10 subjects × 10 classes) |
| Early Stopping | Patience = 7 epochs |
""")

st.markdown("### 🌍 Applications")
st.markdown("""
- **🏠 Smart Home Control** — Control lights, temperature, and appliances with gestures
- **🎮 Gaming Interaction** — Natural gesture-based game controls  
- **📽️ Virtual Presentations** — Navigate slides with hand gestures
- **♿ Accessibility Systems** — Hands-free computer interaction for users with disabilities
- **🤖 IoT Control** — Gesture-based commands for connected devices
- **🔬 HCI Research** — Advancing human-computer interaction research
""")

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,.4);font-size:.85rem">
    <strong>GestureAI v2.0</strong> — Built with ❤️ using MediaPipe + TensorFlow + Streamlit<br>
    © 2026 Hand Gesture Recognition System
</div>
""", unsafe_allow_html=True)
