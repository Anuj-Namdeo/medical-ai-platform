"""
cnn_model.py
============
Custom Convolutional Neural Network for Chest X-Ray Pneumonia Detection.

Architecture:
  4 × ConvBlock (Conv2D → BatchNorm → ReLU → MaxPool)
  GlobalAveragePooling
  Dense(512) → Dropout → Dense(256) → Dropout → Dense(1, sigmoid)

Includes:
  - Training pipeline with callbacks
  - Evaluation pipeline
  - Model saving/loading
  - Training history visualization
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, TensorBoard, CSVLogger
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple, Optional

from src.logger import get_logger

logger = get_logger(__name__)
os.makedirs("./models/saved_models",  exist_ok=True)
os.makedirs("./models/checkpoints",   exist_ok=True)
os.makedirs("./models/model_weights", exist_ok=True)
os.makedirs("./logs",                 exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CNN Architecture
# ─────────────────────────────────────────────────────────────────────────────
def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    dropout_rate: float = 0.5,
    l2_reg: float = 0.001
) -> keras.Model:
    """
    Build custom CNN for binary classification.

    Architecture designed specifically for chest X-ray analysis:
    - 4 ConvBlocks capture features at increasing abstraction levels
    - BatchNormalization stabilizes training
    - GlobalAveragePooling reduces overfitting vs Flatten
    - Dropout regularization prevents memorization
    - L2 regularization on Dense layers

    Args:
        input_shape (tuple): (height, width, channels)
        num_classes (int): 1 for binary, N for multiclass
        dropout_rate (float): Dropout probability
        l2_reg (float): L2 regularization factor

    Returns:
        keras.Model: Compiled CNN model
    """
    inputs = keras.Input(shape=input_shape, name="input_layer")

    # ── Block 1: Low-level features (edges, textures) ──────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_1",
                      kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1_2",
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    # ── Block 2: Mid-level features (shapes, patterns) ─────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_1",
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2_2",
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # ── Block 3: High-level features (lung structures) ─────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_1",
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3_2",
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.35, name="drop3")(x)

    # ── Block 4: Abstract features (disease indicators) ────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_1",
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4_2",
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu", name="relu4_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Dropout(0.35, name="drop4")(x)

    # ── Classifier Head ────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(512, name="dense1",
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Activation("relu", name="relu_dense1")(x)
    x = layers.Dropout(dropout_rate, name="drop_dense1")(x)

    x = layers.Dense(256, name="dense2",
                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization(name="bn_dense2")(x)
    x = layers.Activation("relu", name="relu_dense2")(x)
    x = layers.Dropout(dropout_rate * 0.8, name="drop_dense2")(x)

    # Output layer: sigmoid for binary classification
    outputs = layers.Dense(
        num_classes,
        activation="sigmoid",
        name="output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn")

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )

    logger.info(f"Custom CNN built. Parameters: {model.count_params():,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────
def get_callbacks(
    model_name: str,
    monitor: str = "val_loss"
) -> list:
    """
    Build training callbacks for robust training.

    Callbacks:
    - EarlyStopping: stops when val_loss stops improving (prevents overfit)
    - ModelCheckpoint: saves best model weights automatically
    - ReduceLROnPlateau: halves LR when plateau detected
    - TensorBoard: logs training metrics for visualization
    - CSVLogger: saves all metrics to CSV

    Args:
        model_name (str): Name used for file naming
        monitor (str): Metric to monitor for ES and LR callbacks

    Returns:
        list: List of configured Keras callbacks
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path  = f"./models/checkpoints/{model_name}_best.h5"
    log_dir    = f"./logs/tensorboard/{model_name}_{timestamp}"
    csv_path   = f"./logs/{model_name}_training.csv"

    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        # Stop training when val_loss stops improving for 10 epochs
        EarlyStopping(
            monitor=monitor,
            patience=10,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),

        # Save the model when val_loss improves
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
            mode="min"
        ),

        # Reduce LR by 50% when val_loss plateaus for 5 epochs
        ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # TensorBoard visualization
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        ),

        # Save all metrics to CSV for later analysis
        CSVLogger(
            filename=csv_path,
            separator=",",
            append=False
        )
    ]

    logger.info(f"Callbacks created for model: {model_name}")
    return callbacks


# ─────────────────────────────────────────────────────────────────────────────
# Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    model: keras.Model,
    train_generator,
    val_generator,
    model_name: str,
    epochs: int = 50,
    class_weights: Optional[Dict] = None
) -> Tuple[keras.Model, dict]:
    """
    Full training pipeline with callbacks and history tracking.

    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        model_name (str): Model identifier for saving
        epochs (int): Maximum training epochs
        class_weights (dict): Optional class weights for imbalanced data

    Returns:
        tuple: (trained_model, history_dict)
    """
    logger.info(f"Starting training: {model_name}")
    logger.info(f"  Epochs:     {epochs}")
    logger.info(f"  Train steps:{len(train_generator)}")
    logger.info(f"  Val steps:  {len(val_generator)}")

    callbacks = get_callbacks(model_name=model_name)

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    logger.info(f"Training complete: {model_name}")
    logger.info(f"  Best val_accuracy: {max(history.history['val_accuracy']):.4f}")
    logger.info(f"  Best val_loss:     {min(history.history['val_loss']):.4f}")

    return model, history.history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model: keras.Model,
    test_generator,
    model_name: str = "model",
    threshold: float = 0.5
) -> Dict:
    """
    Complete model evaluation with all clinical metrics.

    Why Recall is CRITICAL in Medical Diagnosis:
    ─────────────────────────────────────────────
    In disease detection, a False Negative (FN) means a sick patient is told
    they are healthy. This is far more dangerous than a False Positive (FP).
    Therefore we optimize for HIGH RECALL (sensitivity), accepting some
    reduction in precision. The cost of missing pneumonia >> cost of
    unnecessary follow-up.

    Metrics computed:
    - Accuracy: overall correct predictions
    - Precision: of predicted positives, how many are actually positive
    - Recall (Sensitivity): of actual positives, how many did we catch
    - F1 Score: harmonic mean of precision and recall
    - ROC-AUC: model's ability to discriminate between classes
    - Confusion Matrix: TP, TN, FP, FN breakdown

    Args:
        model: Trained Keras model
        test_generator: Test data generator
        model_name (str): Name for saving plots
        threshold (float): Decision threshold for binary classification

    Returns:
        dict: All evaluation metrics
    """
    logger.info(f"Evaluating model: {model_name}")

    # Get predictions
    test_generator.reset()
    y_pred_proba = model.predict(test_generator, verbose=1)
    y_pred_proba = y_pred_proba.flatten()
    y_pred       = (y_pred_proba >= threshold).astype(int)
    y_true       = test_generator.classes

    # Compute metrics
    acc       = accuracy_score(y_true, y_pred)
    prec      = precision_score(y_true, y_pred, zero_division=0)
    rec       = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_pred_proba)
    cm        = confusion_matrix(y_true, y_pred)
    report    = classification_report(
        y_true, y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True
    )

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {
        "model_name":   model_name,
        "accuracy":     round(float(acc),       4),
        "precision":    round(float(prec),      4),
        "recall":       round(float(rec),       4),
        "f1_score":     round(float(f1),        4),
        "roc_auc":      round(float(roc_auc),   4),
        "specificity":  round(float(specificity), 4),
        "sensitivity":  round(float(sensitivity), 4),
        "true_positive":  int(tp),
        "true_negative":  int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "threshold":    threshold
    }

    # Log results
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Precision:   {prec:.4f}")
    logger.info(f"  Recall:      {rec:.4f}  ← MOST CRITICAL for medical AI")
    logger.info(f"  F1 Score:    {f1:.4f}")
    logger.info(f"  ROC-AUC:     {roc_auc:.4f}")
    logger.info(f"  Sensitivity: {sensitivity:.4f}")
    logger.info(f"  Specificity: {specificity:.4f}")
    logger.info(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Save plots
    _plot_confusion_matrix(cm, model_name)
    _plot_roc_curve(y_true, y_pred_proba, roc_auc, model_name)

    return metrics


def _plot_confusion_matrix(cm: np.ndarray, model_name: str):
    """Save confusion matrix heatmap."""
    os.makedirs("./logs/plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NORMAL", "PNEUMONIA"],
        yticklabels=["NORMAL", "PNEUMONIA"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    ax.set_ylabel("Actual Label",    fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    path = f"./logs/plots/{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {path}")


def _plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    roc_auc: float,
    model_name: str
):
    """Save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--",
            label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    path = f"./logs/plots/{model_name}_roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved: {path}")


def plot_training_history(history: dict, model_name: str):
    """
    Plot training/validation accuracy and loss curves.

    Args:
        history (dict): Keras history.history dictionary
        model_name (str): Model name for file naming
    """
    os.makedirs("./logs/plots", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history["accuracy"],     label="Train Accuracy", color="blue")
    axes[0].plot(history["val_accuracy"], label="Val Accuracy",   color="orange")
    axes[0].set_title(f"Training Accuracy — {model_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history["loss"],     label="Train Loss", color="blue")
    axes[1].plot(history["val_loss"], label="Val Loss",   color="orange")
    axes[1].set_title(f"Training Loss — {model_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"./logs/plots/{model_name}_training_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training history plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model Saving & Loading
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model: keras.Model, model_name: str) -> str:
    """
    Save model in H5 format and save metadata JSON.

    Args:
        model: Trained Keras model
        model_name (str): Model identifier

    Returns:
        str: Path where model was saved
    """
    save_path = f"./models/saved_models/{model_name}.h5"
    model.save(save_path)
    logger.info(f"Model saved: {save_path}")

    # Save metadata
    metadata = {
        "model_name":    model_name,
        "saved_at":      datetime.now().isoformat(),
        "parameters":    model.count_params(),
        "input_shape":   str(model.input_shape),
        "output_shape":  str(model.output_shape)
    }
    meta_path = f"./models/saved_models/{model_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return save_path


def load_model(model_name: str) -> keras.Model:
    """
    Load a saved model from disk.

    Args:
        model_name (str): Model identifier

    Returns:
        keras.Model: Loaded model
    """
    model_path = f"./models/saved_models/{model_name}.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = keras.models.load_model(model_path)
    logger.info(f"Model loaded: {model_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from src.preprocessing import build_data_generators, compute_class_weights

    with open("./config/config.yaml", "r", encoding="utf-8") as f:
     config = yaml.safe_load(f)

    # Build generators
    train_gen, val_gen, test_gen = build_data_generators(
        dataset_path=config["data"]["dataset_path"],
        image_size=tuple(config["image"]["target_size"]),
        batch_size=config["training"]["batch_size"]
    )

    # Compute class weights
    class_weights = compute_class_weights(train_gen)

    # Build model
    model = build_custom_cnn(
        input_shape=(*config["image"]["target_size"], 3)
    )
    model.summary()

    # Train
    model, history = train_model(
        model=model,
        train_generator=train_gen,
        val_generator=val_gen,
        model_name="custom_cnn",
        epochs=config["training"]["epochs"],
        class_weights=class_weights
    )

    # Plot history
    plot_training_history(history, "custom_cnn")

    # Evaluate
    metrics = evaluate_model(model, test_gen, model_name="custom_cnn")

    # Save
    save_model(model, "custom_cnn")

    logger.info("CNN training pipeline COMPLETE ✓")
