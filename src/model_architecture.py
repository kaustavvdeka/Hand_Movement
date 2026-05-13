"""
CNN Model Architecture for Gesture Classification.
Optimized Sequential model for TF 2.20 / Python 3.13 compatibility.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_gesture_cnn(
    input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
    num_classes=len(config.GESTURE_CLASSES),
    dropout_rate=config.DROPOUT_RATE
):
    """
    Build a CNN architecture for gesture classification.
    
    Architecture:
        - 3 Convolutional blocks with BatchNorm and MaxPooling
        - Global Average Pooling for spatial invariance
        - Dense classifier head with dropout
        - Softmax output for multi-class classification
    """
    model = models.Sequential(name="GestureCNN")
    model.add(layers.Input(shape=input_shape))

    # Block 1: 32 filters
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2: 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3: 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())

    # Classifier Head
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax', name="gesture_output"))

    return model


def compile_model(model, learning_rate=config.LEARNING_RATE):
    """Compile the model with Adam optimizer and sparse categorical crossentropy."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(model_path=config.MODEL_PATH):
    """Get training callbacks for checkpointing and early stopping."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]


def get_model_summary(model):
    """Return model summary as a string."""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)
