# 🖐️ GestureAI — Hand Gesture Recognition System

A premium, AI-powered computer vision platform designed for real-time hand detection, tracking, and gesture classification. Built with **MediaPipe**, **TensorFlow/Keras**, and an interactive **Streamlit** dashboard.

## 🌟 Features

*   **📷 Live Detection:** Real-time gesture recognition using your webcam with interactive holographic-style landmark rendering.
*   **🖼️ Image Upload:** Analyze static images to detect hands, draw landmarks, and predict the gesture.
*   **🎬 Video Upload:** Process pre-recorded videos frame-by-frame and visualize dynamic tracking.
*   **📊 Analytics Dashboard:** Review model performance, training curves, confusion matrices, and detailed classification metrics.
*   **🧠 Deep Learning Pipeline:** Custom CNN architecture trained for high accuracy (`~99.9%`) on the LeapGestRecog dataset, combined with MediaPipe's spatial landmark extraction for background-invariant Region of Interest (ROI) cropping.

## 🛠️ Technology Stack

*   **Frontend:** Streamlit
*   **Computer Vision:** OpenCV (`cv2`), MediaPipe Tasks API (`HandLandmarker`)
*   **Deep Learning:** TensorFlow 2.x, Keras
*   **Data Processing:** NumPy, Pandas, Scikit-Learn
*   **Visualization:** Matplotlib, Seaborn, Plotly

## 🚀 Quick Start Guide

### 1. Installation & Setup

Clone the repository and install the required dependencies:

```bash
# It is recommended to use a virtual environment
pip install -r requirements.txt
```

### 2. Download the MediaPipe Model

The system uses the modern MediaPipe Tasks API. Ensure you have the `hand_landmarker.task` file inside the `models/` directory.

```bash
mkdir -p models
curl -o models/hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

*(Note: The LeapGestRecog dataset and `.npy` cache files are not included in the repository due to size constraints. If you wish to retrain the model, you must download the dataset into the `archive/` folder).*

### 3. Run the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```
This will start the local server and automatically open the application in your default web browser at `http://localhost:8501`.

## 🧠 Methodology & Architecture

1.  **Hand Detection (MediaPipe):** The input frame is processed by MediaPipe to detect the hand and extract **21 3D spatial landmarks**.
2.  **ROI Extraction:** A tight bounding box is calculated from the landmarks to crop out only the hand, entirely discarding background noise.
3.  **Gesture Classification (CNN):** The cropped Region of Interest (ROI) is resized to `64x64`, converted to grayscale, and fed into a custom Convolutional Neural Network.
4.  **Prediction:** The CNN outputs a probability distribution across 10 distinct gesture classes (e.g., Palm, Fist, Thumbs Up, Peace, etc.).

## 📁 Repository Structure

```text
📂 Skillcraft/
├── app.py                  # Main Streamlit dashboard application
├── train_model.py          # Script for preprocessing data and training the CNN
├── evaluate_model.py       # Script for generating metrics and confusion matrices
├── config.py               # Global configurations and hyperparameters
├── requirements.txt        # Python package dependencies
├── 📂 src/
│   ├── data_pipeline.py    # Dataset loading and splitting logic
│   ├── feature_extractor.py# Geometric calculations and feature extraction
│   ├── gesture_classifier.py # CNN Model inference wrappers
│   ├── hand_detector.py    # MediaPipe integration and ROI extraction
│   ├── model_architecture.py # CNN model definition
│   ├── utils.py            # Helper functions (FPS tracking, logging)
│   └── visualizer.py       # Premium bounding box and UI overlay rendering
├── 📂 pages/               # Streamlit multipage application files
│   ├── 1_📷_Live_Detection.py
│   ├── 2_🖼️_Image_Upload.py
│   ├── 3_🎬_Video_Upload.py
│   ├── 4_📊_Analytics.py
│   └── 5_ℹ️_About.py
```

## 📝 License

This project is licensed under the MIT License.
