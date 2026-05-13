"""
Configuration settings for the Hand Gesture Recognition System.
Centralizes all hyperparameters, paths, and constants.
"""
import os

# ─── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive", "leapGestRecog")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

MODEL_PATH = os.path.join(MODEL_DIR, "gesture_cnn.keras")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.npy")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.npy")

# ─── Gesture Classes ─────────────────────────────────────
GESTURE_CLASSES = {
    0: "Palm",
    1: "L-Shape",
    2: "Fist",
    3: "Fist Moved",
    4: "Thumbs Up",
    5: "Index Pointing",
    6: "OK Sign",
    7: "Palm Moved",
    8: "C-Shape",
    9: "Down",
}

GESTURE_EMOJIS = {
    "Palm": "🖐️",
    "L-Shape": "🤙",
    "Fist": "✊",
    "Fist Moved": "👊",
    "Thumbs Up": "👍",
    "Index Pointing": "☝️",
    "OK Sign": "👌",
    "Palm Moved": "🤚",
    "C-Shape": "🫳",
    "Down": "👇",
}

GESTURE_COMMANDS = {
    "Palm": "Stop / Pause",
    "L-Shape": "Call / Connect",
    "Fist": "Grab / Select",
    "Fist Moved": "Drag / Move",
    "Thumbs Up": "Confirm / Like",
    "Index Pointing": "Point / Click",
    "OK Sign": "Approve / OK",
    "Palm Moved": "Swipe / Navigate",
    "C-Shape": "Grip / Adjust",
    "Down": "Volume Down / Dismiss",
}

# ─── Image Settings ───────────────────────────────────────
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 1  # Grayscale

# ─── Model Hyperparameters ────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
DROPOUT_RATE = 0.4

# ─── MediaPipe Settings ──────────────────────────────────
MP_MAX_HANDS = 2
MP_DETECTION_CONFIDENCE = 0.7
MP_TRACKING_CONFIDENCE = 0.6

# ─── UI Settings ─────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
MAX_HISTORY_LENGTH = 50
