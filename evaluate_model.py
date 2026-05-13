"""
Model Evaluation Script — Detailed metrics, confusion matrix, and analysis.

Usage:
    python evaluate_model.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def evaluate():
    """Run detailed evaluation and generate visualizations."""
    print("=" * 60)
    print("📊 Hand Gesture Recognition — Model Evaluation")
    print("=" * 60)

    # Load test predictions
    pred_path = os.path.join(config.MODEL_DIR, "test_predictions.npy")
    if not os.path.exists(pred_path):
        print("❌ No test predictions found. Run train_model.py first.")
        return

    data = np.load(pred_path, allow_pickle=True).item()
    y_true = data['y_true']
    y_pred = data['y_pred']
    y_prob = data['y_prob']

    class_names = [config.GESTURE_CLASSES[i] for i in range(len(config.GESTURE_CLASSES))]

    # 1. Classification Report
    print("\n📋 Classification Report:")
    print("-" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # 2. Overall Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"\n🎯 Overall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor='gray'
    )
    plt.title('Confusion Matrix — Hand Gesture Recognition', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Gesture', fontsize=12)
    plt.ylabel('True Gesture', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = os.path.join(config.MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n📊 Confusion matrix saved to {cm_path}")

    # 4. Training History Curves
    hist_path = config.HISTORY_PATH
    if os.path.exists(hist_path):
        history = np.load(hist_path, allow_pickle=True).item()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy curve
        axes[0].plot(history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss curve
        axes[1].plot(history['loss'], label='Train', linewidth=2)
        axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = os.path.join(config.MODEL_DIR, "training_curves.png")
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"📈 Training curves saved to {curves_path}")

    # 5. Per-class accuracy
    print("\n📊 Per-class Accuracy:")
    print("-" * 40)
    for i, name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).sum() / mask.sum()
            print(f"  {name:15s}: {class_acc:.4f} ({class_acc * 100:.2f}%)")

    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
