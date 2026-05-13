"""
Utility functions for the Hand Gesture Recognition System.
"""
import cv2
import numpy as np
from datetime import datetime
import time


class FPSCounter:
    """Tracks frames per second using a rolling average."""

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.timestamps = []

    def tick(self):
        """Record a frame timestamp."""
        self.timestamps.append(time.time())
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

    def get_fps(self):
        """Calculate current FPS."""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


class GestureHistory:
    """Tracks gesture detection history with timestamps."""

    def __init__(self, max_length=50):
        self.max_length = max_length
        self.history = []

    def add(self, prediction):
        """Add a prediction to history."""
        entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'gesture': prediction.get('class_name', 'Unknown'),
            'confidence': prediction.get('confidence', 0.0),
            'emoji': prediction.get('emoji', '❓'),
            'command': prediction.get('command', ''),
        }
        self.history.insert(0, entry)
        if len(self.history) > self.max_length:
            self.history.pop()

    def get_recent(self, n=10):
        """Get the n most recent entries."""
        return self.history[:n]

    def get_stats(self):
        """Get gesture frequency statistics."""
        if not self.history:
            return {}
        counts = {}
        for entry in self.history:
            g = entry['gesture']
            counts[g] = counts.get(g, 0) + 1
        return counts

    def clear(self):
        """Clear all history."""
        self.history = []


def preprocess_uploaded_image(image_bytes, target_size=(64, 64)):
    """
    Preprocess an uploaded image for model inference.
    
    Args:
        image_bytes: raw bytes of the uploaded image
        target_size: (height, width) tuple
        
    Returns:
        original BGR image, preprocessed grayscale array
    """
    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, None

    # Create grayscale version for model
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    preprocessed = gray_resized.astype(np.float32) / 255.0
    preprocessed = preprocessed.reshape(1, target_size[0], target_size[1], 1)

    return img, preprocessed


def preprocess_frame_for_model(frame, target_size=(64, 64)):
    """
    Preprocess a BGR video frame for model inference (whole frame).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    preprocessed = gray_resized.astype(np.float32) / 255.0
    preprocessed = preprocessed.reshape(1, target_size[0], target_size[1], 1)
    return preprocessed
