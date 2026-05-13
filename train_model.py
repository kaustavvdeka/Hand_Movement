"""
Model Training Script — Train the CNN gesture classifier.

Two-phase approach:
  1. python train_model.py --preprocess    (load images -> save npy)
  2. python train_model.py --train         (load npy -> train model)
  Or just: python train_model.py           (runs both phases)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import load_dataset, split_dataset


def preprocess():
    """Load images from disk and save as npy arrays."""
    print("\n📁 Loading dataset...")
    images, labels, class_names = load_dataset()

    unique, counts = np.unique(labels, return_counts=True)
    print("\n📊 Class distribution:")
    for idx, count in zip(unique, counts):
        print(f"  {class_names[idx]:15s}: {count:5d} samples")

    print("\n✂️  Splitting dataset...")
    train_data, val_data, test_data = split_dataset(images, labels)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    np.save(os.path.join(config.MODEL_DIR, 'X_train.npy'), train_data[0])
    np.save(os.path.join(config.MODEL_DIR, 'y_train.npy'), train_data[1])
    np.save(os.path.join(config.MODEL_DIR, 'X_val.npy'), val_data[0])
    np.save(os.path.join(config.MODEL_DIR, 'y_val.npy'), val_data[1])
    np.save(os.path.join(config.MODEL_DIR, 'X_test.npy'), test_data[0])
    np.save(os.path.join(config.MODEL_DIR, 'y_test.npy'), test_data[1])
    np.save(config.LABEL_MAP_PATH, class_names, allow_pickle=True)

    print("✅ Data preprocessed and saved to npy files.")
    return class_names


def train_model(class_names=None):
    """Train the CNN model from saved npy data."""
    import tensorflow as tf
    from src.model_architecture import build_gesture_cnn, compile_model, get_model_summary

    print("\n📦 Loading preprocessed data...")
    X_train = np.load(os.path.join(config.MODEL_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(config.MODEL_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(config.MODEL_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(config.MODEL_DIR, 'y_val.npy'))
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}")

    if class_names is None:
        class_names = list(np.load(config.LABEL_MAP_PATH, allow_pickle=True))

    print("\n🏗️  Building CNN model...")
    model = build_gesture_cnn()
    model = compile_model(model)
    print(get_model_summary(model))

    print("\n🚀 Starting training...")
    sys.stdout.flush()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluate
    print("\n📈 Evaluating on test set...")
    X_test = np.load(os.path.join(config.MODEL_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(config.MODEL_DIR, 'y_test.npy'))
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Save history
    np.save(config.HISTORY_PATH, {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'class_names': class_names,
    }, allow_pickle=True)

    # Save test predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    np.save(
        os.path.join(config.MODEL_DIR, "test_predictions.npy"),
        {'y_true': y_test, 'y_pred': y_pred_classes, 'y_prob': y_pred},
        allow_pickle=True
    )

    print(f"\n✅ Training Complete! Test Accuracy: {test_acc * 100:.2f}%")
    print(f"   Model saved to: {config.MODEL_PATH}")


def main():
    print("=" * 60)
    print("🖐️  Hand Gesture Recognition — Model Training")
    print("=" * 60)

    mode = sys.argv[1] if len(sys.argv) > 1 else '--all'

    if mode in ('--preprocess', '--all'):
        class_names = preprocess()
    else:
        class_names = None

    if mode in ('--train', '--all'):
        train_model(class_names)

    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
