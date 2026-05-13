"""
Visualizer — Premium rendering utilities for hand gesture visualization.
Provides overlays, charts, and visual elements for the Streamlit UI.
"""
import cv2
import numpy as np


def draw_prediction_overlay(frame, prediction, bbox=None, hand_idx=0):
    """
    Draw a premium prediction overlay on the frame.
    Shows gesture name, emoji, confidence, and command.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    class_name = prediction.get('class_name', 'Unknown')
    confidence = prediction.get('confidence', 0.0)
    emoji = prediction.get('emoji', '❓')
    command = prediction.get('command', '')

    # Semi-transparent background panel
    panel_h = 100
    panel_y = h - panel_h - 10 - (hand_idx * 110)
    cv2.rectangle(overlay, (10, panel_y), (w - 10, panel_y + panel_h), (20, 20, 30), -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Confidence bar
    bar_width = int((w - 40) * confidence)
    bar_color = _confidence_color(confidence)
    cv2.rectangle(frame, (20, panel_y + 75), (20 + bar_width, panel_y + 90), bar_color, -1)
    cv2.rectangle(frame, (20, panel_y + 75), (w - 20, panel_y + 90), (80, 80, 100), 1)

    # Text
    text = f"{class_name}"
    cv2.putText(frame, text, (25, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    conf_text = f"{confidence * 100:.1f}%"
    cv2.putText(frame, conf_text, (w - 100, panel_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

    cmd_text = f"Command: {command}"
    cv2.putText(frame, cmd_text, (25, panel_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 200), 1)

    # Bounding box with prediction label
    if bbox:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bar_color, 2)

        # Label background
        label = f"{class_name} {confidence * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x_min, y_min - th - 10), (x_min + tw + 10, y_min), bar_color, -1)
        cv2.putText(frame, label, (x_min + 5, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def draw_fps_counter(frame, fps):
    """Draw FPS counter on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 170), 2)
    return frame


def draw_no_hand_message(frame):
    """Draw a message when no hand is detected."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 4, h // 2 - 30), (3 * w // 4, h // 2 + 30), (20, 20, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.putText(frame, "Show your hand to the camera", (w // 4 + 15, h // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 170), 2)
    return frame


def _confidence_color(confidence):
    """Return a color based on confidence level (BGR)."""
    if confidence >= 0.85:
        return (0, 230, 118)    # Green
    elif confidence >= 0.65:
        return (0, 200, 255)    # Amber
    elif confidence >= 0.45:
        return (0, 165, 255)    # Orange
    else:
        return (0, 80, 255)     # Red


def create_landmark_visualization(landmarks, frame_size=(400, 400)):
    """
    Create a clean standalone visualization of hand landmarks.
    
    Args:
        landmarks: list of (x, y, z) normalized landmarks
        frame_size: output image size
        
    Returns:
        numpy array image of the landmark visualization
    """
    h, w = frame_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 30)  # Dark background

    if not landmarks:
        return canvas

    # Scale landmarks to canvas
    points = []
    for (x, y, z) in landmarks:
        px = int(x * w * 0.8 + w * 0.1)
        py = int(y * h * 0.8 + h * 0.1)
        points.append((px, py))

    # Connections (MediaPipe hand topology)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # Index
        (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),               # Palm
    ]

    # Draw connections
    for (i, j) in connections:
        if i < len(points) and j < len(points):
            cv2.line(canvas, points[i], points[j], (0, 180, 230), 2, cv2.LINE_AA)

    # Draw landmarks
    for idx, pt in enumerate(points):
        # Fingertips in bright color
        if idx in [4, 8, 12, 16, 20]:
            cv2.circle(canvas, pt, 6, (0, 255, 170), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 8, (0, 255, 170), 1, cv2.LINE_AA)
        elif idx == 0:
            cv2.circle(canvas, pt, 7, (255, 100, 100), -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, pt, 4, (0, 200, 255), -1, cv2.LINE_AA)

    return canvas
