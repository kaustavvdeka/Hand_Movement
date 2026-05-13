"""
Data Pipeline — Loading, Preprocessing & Augmentation
Handles the LeapGestRecog dataset with train/val/test splitting.
Uses numpy arrays directly for compatibility.
"""
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_dataset(data_dir=None, img_size=(config.IMG_HEIGHT, config.IMG_WIDTH)):
    """
    Load images from the LeapGestRecog dataset.
    Directory structure: data_dir/subject/gesture_class/images.png
    
    Returns:
        images: np.ndarray of shape (N, H, W, 1) normalized to [0, 1]
        labels: np.ndarray of integer labels
        class_names: list of class name strings
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    images = []
    labels = []

    # Map folder names to class indices
    gesture_folders = sorted([
        "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
        "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"
    ])
    class_names = [config.GESTURE_CLASSES[i] for i in range(len(gesture_folders))]

    print(f"Loading dataset from: {data_dir}")
    print(f"Classes: {class_names}")

    # Iterate through each subject folder (00-09)
    subject_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
    ])

    for subject in subject_dirs:
        subject_path = os.path.join(data_dir, subject)
        for class_idx, gesture_folder in enumerate(gesture_folders):
            gesture_path = os.path.join(subject_path, gesture_folder)
            if not os.path.exists(gesture_path):
                continue

            for img_file in os.listdir(gesture_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                img_path = os.path.join(gesture_path, img_file)
                try:
                    img = Image.open(img_path).convert('L')  # Grayscale
                    img = img.resize(img_size, Image.BILINEAR)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"  Skipping {img_path}: {e}")

    images = np.array(images).reshape(-1, img_size[0], img_size[1], 1)
    labels = np.array(labels, dtype=np.int32)

    print(f"Loaded {len(images)} images across {len(class_names)} classes")
    return images, labels, class_names


def split_dataset(images, labels, val_split=config.VALIDATION_SPLIT, test_split=config.TEST_SPLIT, seed=42):
    """
    Split dataset into train, validation, and test sets with stratification.
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_split, random_state=seed, stratify=labels
    )

    # Second split: separate validation from training
    val_ratio = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=seed, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_augmented_generator(X, y, batch_size=config.BATCH_SIZE):
    """
    Create a data generator with augmentation using numpy operations.
    Yields (batch_images, batch_labels) with random transformations.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
    )
    
    return datagen.flow(X, y, batch_size=batch_size, shuffle=True)
