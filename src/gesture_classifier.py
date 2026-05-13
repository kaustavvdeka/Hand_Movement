"""
Gesture Classifier — Model inference for gesture recognition.
Loads the trained CNN model and performs predictions.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GestureClassifier:
    """
    Loads a trained CNN model and performs gesture classification.
    Handles preprocessing, inference, and post-processing.
    """

    def __init__(self, model_path=None):
        self.model = None
        self.model_loaded = False
        self.class_names = [config.GESTURE_CLASSES[i] for i in range(len(config.GESTURE_CLASSES))]

        if model_path is None:
            model_path = config.MODEL_PATH

        self._load_model(model_path)

    def _load_model(self, model_path):
        """Load the trained Keras model."""
        if os.path.exists(model_path):
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                self.model_loaded = False
        else:
            print(f"⚠️ Model not found at {model_path}. Train the model first.")
            self.model_loaded = False

    def predict(self, preprocessed_image):
        """
        Predict gesture class from a preprocessed image.
        
        Args:
            preprocessed_image: numpy array of shape (1, H, W, 1), normalized
            
        Returns:
            dict: {
                'class_name': str,
                'class_index': int,
                'confidence': float,
                'all_probabilities': dict[str, float],
                'emoji': str,
                'command': str
            }
        """
        if not self.model_loaded or self.model is None:
            return self._fallback_prediction()

        try:
            predictions = self.model.predict(preprocessed_image, verbose=0)
            probs = predictions[0]

            class_index = int(np.argmax(probs))
            confidence = float(probs[class_index])
            class_name = self.class_names[class_index]

            all_probs = {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            }

            return {
                'class_name': class_name,
                'class_index': class_index,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'emoji': config.GESTURE_EMOJIS.get(class_name, "🖐️"),
                'command': config.GESTURE_COMMANDS.get(class_name, "Unknown"),
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction()

    def predict_batch(self, images):
        """Predict gestures for a batch of preprocessed images."""
        if not self.model_loaded or self.model is None:
            return [self._fallback_prediction() for _ in range(len(images))]

        predictions = self.model.predict(images, verbose=0)
        results = []
        for probs in predictions:
            class_index = int(np.argmax(probs))
            confidence = float(probs[class_index])
            class_name = self.class_names[class_index]
            results.append({
                'class_name': class_name,
                'class_index': class_index,
                'confidence': confidence,
                'all_probabilities': {
                    self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))
                },
                'emoji': config.GESTURE_EMOJIS.get(class_name, "🖐️"),
                'command': config.GESTURE_COMMANDS.get(class_name, "Unknown"),
            })
        return results

    def _fallback_prediction(self):
        """Return a default prediction when model is not available."""
        return {
            'class_name': "No Model",
            'class_index': -1,
            'confidence': 0.0,
            'all_probabilities': {},
            'emoji': "❓",
            'command': "Train the model first",
        }

    def is_ready(self):
        """Check if the classifier is ready for predictions."""
        return self.model_loaded
