"""
📊 Analytics — Model performance metrics and evaluation dashboard.
"""
import streamlit as st
import numpy as np
import os, sys
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title="Analytics — GestureAI", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
.stApp{font-family:'Inter',sans-serif}
#MainMenu,footer,header{visibility:hidden}
.page-header{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);border-radius:16px;padding:2rem;margin-bottom:1.5rem;border:1px solid rgba(255,255,255,.06)}
.page-title{font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#00ffaa,#ff6b6b);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.metric-card{background:linear-gradient(145deg,rgba(25,25,45,.95),rgba(15,15,35,.98));border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:1.5rem;text-align:center}
.metric-value{font-size:2rem;font-weight:800;background:linear-gradient(135deg,#00ffaa,#00c8ff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.metric-label{font-size:.85rem;color:rgba(255,255,255,.5);margin-top:.3rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <div class="page-title">📊 Model Analytics</div>
    <div style="color:rgba(255,255,255,.6);font-size:.9rem;margin-top:.3rem">
        Comprehensive performance metrics, training curves, and confusion matrix analysis.
    </div>
</div>
""", unsafe_allow_html=True)

# Load data
hist_path = config.HISTORY_PATH
pred_path = os.path.join(config.MODEL_DIR, "test_predictions.npy")
has_history = os.path.exists(hist_path)
has_preds = os.path.exists(pred_path)

if not has_history and not has_preds:
    st.warning("⚠️ No training data found. Run `python train_model.py` and `python evaluate_model.py` first.")
    st.stop()

# ─── Metrics Row ─────────────────────────────────────
if has_history:
    history = np.load(hist_path, allow_pickle=True).item()
    test_acc = history.get('test_accuracy', 0)
    test_loss = history.get('test_loss', 0)
    best_val = max(history.get('val_accuracy', [0]))
    epochs_trained = len(history.get('accuracy', []))

    cols = st.columns(4)
    metrics = [
        ("Test Accuracy", f"{test_acc*100:.1f}%"),
        ("Test Loss", f"{test_loss:.4f}"),
        ("Best Val Acc", f"{best_val*100:.1f}%"),
        ("Epochs", str(epochs_trained)),
    ]
    for col, (label, val) in zip(cols, metrics):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Training curves
    tab1, tab2, tab3 = st.tabs(["📈 Training Curves", "🔥 Confusion Matrix", "📋 Per-Class Metrics"])

    with tab1:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
        fig.add_trace(go.Scatter(y=history['accuracy'], name='Train Acc', line=dict(color='#00ffaa', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(y=history['val_accuracy'], name='Val Acc', line=dict(color='#00c8ff', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(y=history['loss'], name='Train Loss', line=dict(color='#ff6b6b', width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss', line=dict(color='#ffa726', width=2, dash='dash')), row=1, col=2)
        fig.update_layout(
            plot_bgcolor='rgba(15,15,35,0.95)', paper_bgcolor='rgba(0,0,0,0)',
            font_color='rgba(255,255,255,0.7)', height=400,
            legend=dict(orientation="h", y=-0.15)
        )
        fig.update_xaxes(title_text="Epoch", gridcolor='rgba(255,255,255,0.05)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if has_preds:
            from sklearn.metrics import confusion_matrix
            data = np.load(pred_path, allow_pickle=True).item()
            y_true, y_pred = data['y_true'], data['y_pred']
            class_names = [config.GESTURE_CLASSES[i] for i in range(len(config.GESTURE_CLASSES))]
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=class_names, y=class_names, color_continuous_scale="YlOrRd",
                             text_auto=True, aspect="auto")
            fig_cm.update_layout(
                plot_bgcolor='rgba(15,15,35,0.95)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='rgba(255,255,255,0.7)', height=500,
                title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Run `python evaluate_model.py` to generate predictions.")

    with tab3:
        if has_preds:
            from sklearn.metrics import classification_report
            data = np.load(pred_path, allow_pickle=True).item()
            y_true, y_pred = data['y_true'], data['y_pred']
            class_names = [config.GESTURE_CLASSES[i] for i in range(len(config.GESTURE_CLASSES))]
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

            import pandas as pd
            df = pd.DataFrame(report).T
            df = df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
            df = df.round(4)

            st.dataframe(df.style.background_gradient(cmap='YlGn', subset=['precision', 'recall', 'f1-score']),
                        use_container_width=True)

            # Per-class accuracy bar chart
            per_class_acc = []
            for i, name in enumerate(class_names):
                mask = y_true == i
                if mask.sum() > 0:
                    acc = (y_pred[mask] == i).sum() / mask.sum()
                    per_class_acc.append({'Gesture': name, 'Accuracy': acc,
                                         'Emoji': config.GESTURE_EMOJIS.get(name, '')})
            df_acc = pd.DataFrame(per_class_acc)
            fig_bar = px.bar(df_acc, x='Gesture', y='Accuracy', color='Accuracy',
                           color_continuous_scale='Emrld', text_auto='.1%')
            fig_bar.update_layout(
                plot_bgcolor='rgba(15,15,35,0.95)', paper_bgcolor='rgba(0,0,0,0)',
                font_color='rgba(255,255,255,0.7)', height=400, title="Per-Class Accuracy"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Run `python evaluate_model.py` to generate metrics.")

# Model architecture info
with st.expander("🏗️ Model Architecture"):
    st.markdown("""
    ```
    GestureCNN Architecture:
    ┌─────────────────────────────────────┐
    │ Input: 64×64×1 (Grayscale)          │
    ├─────────────────────────────────────┤
    │ Conv Block 1: 32 filters, BN, ReLU  │
    │ Conv Block 1: 32 filters, BN, ReLU  │
    │ MaxPool 2×2 + Dropout 0.25          │
    ├─────────────────────────────────────┤
    │ Conv Block 2: 64 filters, BN, ReLU  │
    │ Conv Block 2: 64 filters, BN, ReLU  │
    │ MaxPool 2×2 + Dropout 0.25          │
    ├─────────────────────────────────────┤
    │ Conv Block 3: 128 filters, BN, ReLU │
    │ Conv Block 3: 128 filters, BN, ReLU │
    │ MaxPool 2×2 + Dropout 0.30          │
    ├─────────────────────────────────────┤
    │ Conv Block 4: 256 filters, BN, ReLU │
    │ Global Average Pooling              │
    ├─────────────────────────────────────┤
    │ Dense 256 + BN + ReLU + Dropout 0.4 │
    │ Dense 128 + BN + ReLU + Dropout 0.4 │
    │ Dense 10 + Softmax                  │
    └─────────────────────────────────────┘
    ```
    """)
