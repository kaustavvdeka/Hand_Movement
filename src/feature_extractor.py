"""
Feature Extractor — Extracts meaningful features from hand landmarks.
Computes geometric features for gesture analysis.
"""
import numpy as np
import math


class FeatureExtractor:
    """
    Extracts features from MediaPipe hand landmarks for gesture analysis.
    Features include angles, distances, and finger states.
    """

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [3, 6, 10, 14, 18]
    FINGER_MCPS = [2, 5, 9, 13, 17]

    @staticmethod
    def compute_distance(p1, p2):
        """Compute Euclidean distance between two 3D points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    @staticmethod
    def compute_angle(p1, p2, p3):
        """Compute angle at p2 formed by p1-p2-p3 in degrees."""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def get_finger_states(self, landmarks):
        """
        Determine which fingers are extended (open) or closed.
        
        Returns:
            dict with finger names and their states (True = extended)
        """
        # Thumb: compare tip x-position relative to IP joint
        # (accounts for left/right hand orientation)
        thumb_extended = landmarks[self.THUMB_TIP][0] < landmarks[self.THUMB_IP][0]

        # For other fingers: tip is above PIP joint (y increases downward)
        finger_names = ['index', 'middle', 'ring', 'pinky']
        finger_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        finger_pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]

        states = {'thumb': thumb_extended}
        for name, tip, pip in zip(finger_names, finger_tips, finger_pips):
            states[name] = landmarks[tip][1] < landmarks[pip][1]

        return states

    def extract_features(self, landmarks):
        """
        Extract a comprehensive feature vector from 21 hand landmarks.
        
        Features:
            - 5 finger extension states (binary)
            - 10 inter-finger distances
            - 5 finger curl angles
            - 5 finger-to-wrist distances
            - Palm size (wrist to middle MCP distance)
            
        Returns:
            numpy array of features (26 features)
        """
        features = []
        
        # 1. Finger states (5 features)
        states = self.get_finger_states(landmarks)
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            features.append(1.0 if states[finger] else 0.0)

        # 2. Finger tip distances from wrist (5 features)
        wrist = landmarks[self.WRIST]
        for tip_idx in self.FINGER_TIPS:
            d = self.compute_distance(landmarks[tip_idx], wrist)
            features.append(d)

        # 3. Inter-finger tip distances (10 features)
        for i in range(len(self.FINGER_TIPS)):
            for j in range(i + 1, len(self.FINGER_TIPS)):
                d = self.compute_distance(
                    landmarks[self.FINGER_TIPS[i]],
                    landmarks[self.FINGER_TIPS[j]]
                )
                features.append(d)

        # 4. Finger curl angles at PIP joints (5 features)
        finger_chains = [
            (self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP),
            (self.INDEX_MCP, self.INDEX_PIP, self.INDEX_TIP),
            (self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_TIP),
            (self.RING_MCP, self.RING_PIP, self.RING_TIP),
            (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_TIP),
        ]
        for mcp, pip, tip in finger_chains:
            angle = self.compute_angle(landmarks[mcp], landmarks[pip], landmarks[tip])
            features.append(angle / 180.0)  # Normalize to [0, 1]

        # 5. Palm size (1 feature)
        palm_size = self.compute_distance(wrist, landmarks[self.MIDDLE_MCP])
        features.append(palm_size)

        return np.array(features, dtype=np.float32)

    def get_feature_names(self):
        """Return names for each feature in the feature vector."""
        names = []
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            names.append(f"{f}_extended")
        for tip_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            names.append(f"{tip_name}_to_wrist_dist")
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i in range(5):
            for j in range(i + 1, 5):
                names.append(f"{finger_names[i]}_{finger_names[j]}_dist")
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            names.append(f"{f}_curl_angle")
        names.append("palm_size")
        return names
