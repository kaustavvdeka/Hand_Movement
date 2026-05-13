"""
Hand Detector — MediaPipe-based hand detection and landmark extraction.
Provides real-time hand tracking with landmark overlay rendering.
Updated for MediaPipe 0.10+ tasks API (Python 3.13 compatibility).
"""
import cv2
import numpy as np
import mediapipe as mp
import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class HandDetector:
    """
    Detects hands in images/video frames using MediaPipe Tasks API.
    Extracts 21 3D landmarks per hand.
    """

    def __init__(
        self,
        max_hands=config.MP_MAX_HANDS,
        detection_confidence=config.MP_DETECTION_CONFIDENCE,
        tracking_confidence=config.MP_TRACKING_CONFIDENCE,
        static_mode=False,
    ):
        self.static_mode = static_mode
        self.last_timestamp = 0
        
        # Determine path to the model file
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            'models', 'hand_landmarker.task'
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hand landmarker model not found at {model_path}. Please download it.")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        running_mode = VisionRunningMode.IMAGE if static_mode else VisionRunningMode.VIDEO

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        Detect hands in a BGR frame.
        
        Returns:
            results: MediaPipe HandLandmarkerResult
            rgb_frame: The converted RGB frame
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        if self.static_mode:
            results = self.landmarker.detect(mp_image)
        else:
            timestamp = int(time.time() * 1000)
            if timestamp <= self.last_timestamp:
                timestamp = self.last_timestamp + 1
            self.last_timestamp = timestamp
            results = self.landmarker.detect_for_video(mp_image, timestamp)
            
        return results, rgb_frame

    def get_landmarks(self, results, frame_shape):
        """
        Extract normalized and pixel landmarks from detection results.
        
        Returns:
            List of dicts, one per detected hand:
            {
                'landmarks': [(x, y, z), ...],  # 21 normalized landmarks
                'pixel_landmarks': [(px, py), ...],  # 21 pixel landmarks
                'handedness': 'Left' or 'Right',
                'bbox': (x_min, y_min, x_max, y_max)  # bounding box in pixels
            }
        """
        hands_data = []
        if not results or not getattr(results, 'hand_landmarks', None):
            return hands_data

        h, w, _ = frame_shape

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            landmarks = []
            pixel_landmarks = []
            x_coords, y_coords = [], []

            for lm in hand_landmarks:
                landmarks.append((lm.x, lm.y, lm.z))
                px, py = int(lm.x * w), int(lm.y * h)
                pixel_landmarks.append((px, py))
                x_coords.append(px)
                y_coords.append(py)

            # Get handedness
            handedness = "Unknown"
            if getattr(results, 'handedness', None) and idx < len(results.handedness):
                handedness = results.handedness[idx][0].category_name

            # Bounding box with padding
            pad = 20
            bbox = (
                max(0, min(x_coords) - pad),
                max(0, min(y_coords) - pad),
                min(w, max(x_coords) + pad),
                min(h, max(y_coords) + pad),
            )

            hands_data.append({
                'landmarks': landmarks,
                'pixel_landmarks': pixel_landmarks,
                'handedness': handedness,
                'bbox': bbox,
            })

        return hands_data

    def draw_landmarks(self, frame, results, draw_bbox=True):
        """
        Draw hand landmarks and connections on the frame manually.
        """
        annotated = frame.copy()

        if not results or not getattr(results, 'hand_landmarks', None):
            return annotated

        hands_data = self.get_landmarks(results, frame.shape)

        for idx, hand_data in enumerate(hands_data):
            px_lms = hand_data['pixel_landmarks']
            
            # Draw connections
            for connection in HAND_CONNECTIONS:
                pt1 = px_lms[connection[0]]
                pt2 = px_lms[connection[1]]
                cv2.line(annotated, pt1, pt2, (0, 200, 255), 2)
                
            # Draw landmarks
            for pt in px_lms:
                cv2.circle(annotated, pt, 3, (0, 255, 170), -1)

            # Draw bounding box
            if draw_bbox:
                bbox = hand_data['bbox']
                handedness = hand_data['handedness']
                cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 170), 2)
                cv2.putText(
                    annotated, handedness,
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 170), 2
                )

        return annotated

    def extract_hand_roi(self, frame, bbox, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH)):
        """
        Extract and preprocess hand region of interest for model inference.
        
        Returns:
            Grayscale, resized, normalized ROI ready for CNN input.
        """
        x_min, y_min, x_max, y_max = bbox
        roi = frame[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return None

        # Convert to grayscale
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Resize
        roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)

        # Normalize
        roi = roi.astype(np.float32) / 255.0

        # Reshape for model: (1, H, W, 1)
        roi = roi.reshape(1, target_size[0], target_size[1], 1)

        return roi

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
