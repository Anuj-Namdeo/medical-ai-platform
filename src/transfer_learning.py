"""
transfer_learning.py
====================
Transfer Learning Models for Chest X-Ray Pneumonia Detection.

Models implemented:
  1. ResNet50   - Deep residual learning with skip connections
  2. DenseNet121 - Densely connected CNN (excellent for medical imaging)
  3. EfficientNetB0 - Compound-scaled network (best accuracy/params ratio)

Strategy:
  Phase 1: Freeze base model → train only classifier head (5 epochs)
  Phase 2: Unfreeze top layers → fine-tune entire network (remaining epochs)

Why Transfer Learning works for medical images:
  ImageNet pretrained models learned universal visual features (edges, textures,
  shapes). These transfer well to X-rays even though the domains differ.
  We only need to adapt the final classification layers.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from typing import Dict, Tuple, Optional, List

from src.cnn_model import (
    get_callbacks, evaluate_model, save_model,
    plot_training_history
)
from src.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base Transfer Learning Builder
# ─────────────────────────────────────────────────────────────────────────────
def _build_transfer_model(
    base_model,
    model_name: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dense_units: List[int] = [512, 256],
    dropout_rate: float = 0.5,
    learning_rate: float = 0.0001
) -> keras.Model:
    """
    Shared builder for all transfer learning models.

    Args:
        base_model: Pretrained Keras application (ResNet50, etc.)
        model_name (str): Name identifier
        input_shape (tuple): Input dimensions
        dense_units (list): Dense layer sizes for classifier head
        dropout_rate (float): Dropout probability
        learning_rate (float): Initial learning rate

    Returns:
        keras.Model: Compiled model with frozen base
    """
    # Freeze all base model layers initially
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape, name="input_layer")

    # Pass through base model (not training)
    x = base_model(inputs, training=False)

    # Global Average Pooling to flatten spatial features
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Classifier head
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_dense_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_dense_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    # Binary output
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    logger.info(f"{model_name} built | Total params: {model.count_params():,} | "
                f"Trainable layers: {trainable}")

    return model


def _unfreeze_and_finetune(
    model: keras.Model,
    base_model,
    unfreeze_from_layer: int,
    learning_rate: float = 0.00001
) -> keras.Model:
    """
    Unfreeze top layers of base model for fine-tuning.

    Fine-tuning makes the base model adapt its features to medical images,
    beyond what it learned on ImageNet. Use a very small learning rate
    to avoid destroying the pretrained weights.

    Args:
        model: Full model (base + head)
        base_model: The pretrained base
        unfreeze_from_layer (int): Layer index from which to unfreeze
        learning_rate (float): Very small LR for fine-tuning

    Returns:
        keras.Model: Recompiled model with some base layers unfrozen
    """
    base_model.trainable = True

    # Freeze layers before unfreeze_from_layer
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    # Unfreeze layers from unfreeze_from_layer onwards
    for layer in base_model.layers[unfreeze_from_layer:]:
        layer.trainable = True

    # Recompile with smaller learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
    )

    trainable_params = sum(
        np.prod(var.shape) for var in model.trainable_variables
    )
    logger.info(f"Fine-tuning enabled from layer {unfreeze_from_layer}. "
                f"Trainable params: {trainable_params:,}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# ResNet50
# ─────────────────────────────────────────────────────────────────────────────
def build_resnet50(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dense_units: List[int] = [512, 256],
    dropout_rate: float = 0.5
) -> Tuple[keras.Model, object]:
    """
    Build ResNet50 transfer learning model.

    ResNet50 Key Features:
    - 50 layers deep with residual/skip connections
    - Skip connections solve vanishing gradient problem
    - 25.6M parameters (ImageNet pretrained)
    - Proven effective for medical image classification

    Args:
        input_shape: Image dimensions
        dense_units: Classifier head layer sizes
        dropout_rate: Dropout probability

    Returns:
        tuple: (full_model, base_model_reference)
    """
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    logger.info(f"ResNet50 base loaded. Total layers: {len(base.layers)}")

    model = _build_transfer_model(
        base_model=base,
        model_name="resnet50",
        input_shape=input_shape,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )

    return model, base


def train_resnet50(
    train_generator,
    val_generator,
    test_generator,
    epochs: int = 30,
    class_weights: Optional[Dict] = None,
    input_shape: Tuple[int, int, int] = (224, 224, 3)
) -> Tuple[keras.Model, Dict, Dict]:
    """
    Two-phase training for ResNet50:
    Phase 1: Train only classifier head (fast convergence)
    Phase 2: Fine-tune top base layers (model adapts to domain)

    Args:
        train_generator: Training data
        val_generator: Validation data
        test_generator: Test data
        epochs (int): Total epochs
        class_weights (dict): Class weights for imbalance
        input_shape (tuple): Image dimensions

    Returns:
        tuple: (trained_model, combined_history, evaluation_metrics)
    """
    model, base_model = build_resnet50(input_shape=input_shape)
    model.summary(print_fn=logger.info)

    # ── Phase 1: Train head only ───────────────────────────────────────────
    logger.info("ResNet50 Phase 1: Training classifier head (base frozen)")
    warmup_epochs = min(5, epochs // 4)

    history_p1 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=warmup_epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        class_weight=class_weights,
        verbose=1
    )

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────
    logger.info("ResNet50 Phase 2: Fine-tuning (unfreezing top layers)")
    model = _unfreeze_and_finetune(model, base_model, unfreeze_from_layer=100)
    callbacks = get_callbacks("resnet50")

    history_p2 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs - warmup_epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Combine histories
    combined_history = {}
    for key in history_p1.history:
        combined_history[key] = (
            history_p1.history[key] + history_p2.history.get(key, [])
        )

    plot_training_history(combined_history, "resnet50")

    # Evaluate
    metrics = evaluate_model(model, test_generator, "resnet50")
    save_model(model, "resnet50")

    return model, combined_history, metrics


# ─────────────────────────────────────────────────────────────────────────────
# DenseNet121
# ─────────────────────────────────────────────────────────────────────────────
def build_densenet121(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dense_units: List[int] = [512, 256],
    dropout_rate: float = 0.5
) -> Tuple[keras.Model, object]:
    """
    Build DenseNet121 transfer learning model.

    DenseNet121 Key Features:
    - Each layer connected to ALL subsequent layers (dense connections)
    - Strong feature reuse → very parameter-efficient
    - 8M parameters (much smaller than ResNet50)
    - Used in CheXNet paper for chest X-ray analysis (Stanford, 2017)
    - EXCELLENT choice for medical imaging

    Args:
        input_shape: Image dimensions
        dense_units: Classifier layer sizes
        dropout_rate: Dropout probability

    Returns:
        tuple: (full_model, base_model_reference)
    """
    base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    logger.info(f"DenseNet121 base loaded. Total layers: {len(base.layers)}")

    model = _build_transfer_model(
        base_model=base,
        model_name="densenet121",
        input_shape=input_shape,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )

    return model, base


def train_densenet121(
    train_generator,
    val_generator,
    test_generator,
    epochs: int = 30,
    class_weights: Optional[Dict] = None,
    input_shape: Tuple[int, int, int] = (224, 224, 3)
) -> Tuple[keras.Model, Dict, Dict]:
    """
    Two-phase training for DenseNet121.

    Returns:
        tuple: (trained_model, combined_history, evaluation_metrics)
    """
    model, base_model = build_densenet121(input_shape=input_shape)

    # Phase 1
    logger.info("DenseNet121 Phase 1: Training head only")
    warmup_epochs = min(5, epochs // 4)

    history_p1 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=warmup_epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        class_weight=class_weights,
        verbose=1
    )

    # Phase 2
    logger.info("DenseNet121 Phase 2: Fine-tuning")
    model = _unfreeze_and_finetune(model, base_model, unfreeze_from_layer=100)
    callbacks = get_callbacks("densenet121")

    history_p2 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs - warmup_epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    combined_history = {}
    for key in history_p1.history:
        combined_history[key] = (
            history_p1.history[key] + history_p2.history.get(key, [])
        )

    plot_training_history(combined_history, "densenet121")
    metrics = evaluate_model(model, test_generator, "densenet121")
    save_model(model, "densenet121")

    return model, combined_history, metrics


# ─────────────────────────────────────────────────────────────────────────────
# EfficientNetB0
# ─────────────────────────────────────────────────────────────────────────────
def build_efficientnet(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dense_units: List[int] = [512, 256],
    dropout_rate: float = 0.5
) -> Tuple[keras.Model, object]:
    """
    Build EfficientNetB0 transfer learning model.

    EfficientNet Key Features:
    - Compound scaling: depth + width + resolution scaled together
    - Best accuracy-to-parameters ratio (2019 Google Brain)
    - B0: 5.3M parameters (smallest of the EfficientNet family)
    - State-of-the-art on ImageNet when published

    Args:
        input_shape: Image dimensions
        dense_units: Classifier layer sizes
        dropout_rate: Dropout probability

    Returns:
        tuple: (full_model, base_model_reference)
    """
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    logger.info(f"EfficientNetB0 base loaded. Total layers: {len(base.layers)}")

    model = _build_transfer_model(
        base_model=base,
        model_name="efficientnetb0",
        input_shape=input_shape,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )

    return model, base


def train_efficientnet(
    train_generator,
    val_generator,
    test_generator,
    epochs: int = 30,
    class_weights: Optional[Dict] = None,
    input_shape: Tuple[int, int, int] = (224, 224, 3)
) -> Tuple[keras.Model, Dict, Dict]:
    """
    Two-phase training for EfficientNetB0.

    Returns:
        tuple: (trained_model, combined_history, evaluation_metrics)
    """
    model, base_model = build_efficientnet(input_shape=input_shape)

    # Phase 1
    logger.info("EfficientNetB0 Phase 1: Training head only")
    warmup_epochs = min(5, epochs // 4)

    history_p1 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=warmup_epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        class_weight=class_weights,
        verbose=1
    )

    # Phase 2
    logger.info("EfficientNetB0 Phase 2: Fine-tuning")
    model = _unfreeze_and_finetune(model, base_model, unfreeze_from_layer=100)
    callbacks = get_callbacks("efficientnetb0")

    history_p2 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs - warmup_epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    combined_history = {}
    for key in history_p1.history:
        combined_history[key] = (
            history_p1.history[key] + history_p2.history.get(key, [])
        )

    plot_training_history(combined_history, "efficientnetb0")
    metrics = evaluate_model(model, test_generator, "efficientnetb0")
    save_model(model, "efficientnetb0")

    return model, combined_history, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison Framework
# ─────────────────────────────────────────────────────────────────────────────
def compare_all_models(
    metrics_list: List[Dict]
) -> None:
    """
    Create comprehensive comparison of all trained models.
    Saves comparison table + multi-metric bar chart.

    Args:
        metrics_list (list): List of metric dicts from evaluate_model()
    """
    import pandas as pd

    comparison_metrics = [
        "model_name", "accuracy", "precision", "recall",
        "f1_score", "roc_auc", "specificity", "sensitivity"
    ]

    rows = []
    for m in metrics_list:
        row = {k: m.get(k, None) for k in comparison_metrics}
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model_name")

    # Print table
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("="*70)
    logger.info("\n" + df.to_string())
    logger.info("\nBest model by Recall (most important for medical AI):")
    best = df["recall"].idxmax()
    logger.info(f"  ✓ {best} with Recall = {df.loc[best, 'recall']:.4f}")

    # Save CSV
    os.makedirs("./logs", exist_ok=True)
    csv_path = "./logs/model_comparison.csv"
    df.to_csv(csv_path)
    logger.info(f"Comparison saved: {csv_path}")

    # Plot comparison
    _plot_model_comparison(df)

    return df


def _plot_model_comparison(df):
    """Save multi-metric bar chart comparing all models."""
    os.makedirs("./logs/plots", exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    x = np.arange(len(df.index))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["steelblue", "darkorange", "green", "red", "purple"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = df[metric].values
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=metric.replace("_", " ").title(),
            color=color,
            alpha=0.85
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xlabel("Model",   fontsize=12)
    ax.set_ylabel("Score",   fontsize=12)
    ax.set_title("Model Performance Comparison — All Metrics", fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df.index, fontsize=11)
    ax.set_ylim([0, 1.12])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = "./logs/plots/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Model comparison chart saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point: Train All Models
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from src.preprocessing import build_data_generators, compute_class_weights
    from src.cnn_model import (
        build_custom_cnn, train_model, evaluate_model as eval_cnn,
        save_model, plot_training_history
    )

    with open("./config/config.yaml", "r", encoding="utf-8") as f:
     config = yaml.safe_load(f)
    img_size    = tuple(config["image"]["target_size"])
    batch_size  = config["training"]["batch_size"]
    epochs      = config["training"]["epochs"]
    dataset_path = config["data"]["dataset_path"]

    # Generators
    train_gen, val_gen, test_gen = build_data_generators(
        dataset_path=dataset_path,
        image_size=img_size,
        batch_size=batch_size
    )
    class_weights = compute_class_weights(train_gen)

    all_metrics = []

    # 1. Custom CNN
    logger.info("\n" + "="*50 + "\nTraining Custom CNN\n" + "="*50)
    cnn = build_custom_cnn(input_shape=(*img_size, 3))
    cnn, cnn_hist = train_model(
        cnn, train_gen, val_gen, "custom_cnn",
        epochs=epochs, class_weights=class_weights
    )
    plot_training_history(cnn_hist, "custom_cnn")
    cnn_metrics = eval_cnn(cnn, test_gen, "custom_cnn")
    save_model(cnn, "custom_cnn")
    all_metrics.append(cnn_metrics)

    # 2. ResNet50
    logger.info("\n" + "="*50 + "\nTraining ResNet50\n" + "="*50)
    _, _, resnet_metrics = train_resnet50(
        train_gen, val_gen, test_gen,
        epochs=epochs, class_weights=class_weights, input_shape=(*img_size, 3)
    )
    all_metrics.append(resnet_metrics)

    # 3. DenseNet121
    logger.info("\n" + "="*50 + "\nTraining DenseNet121\n" + "="*50)
    _, _, densenet_metrics = train_densenet121(
        train_gen, val_gen, test_gen,
        epochs=epochs, class_weights=class_weights, input_shape=(*img_size, 3)
    )
    all_metrics.append(densenet_metrics)

    # 4. EfficientNetB0
    logger.info("\n" + "="*50 + "\nTraining EfficientNetB0\n" + "="*50)
    _, _, efficientnet_metrics = train_efficientnet(
        train_gen, val_gen, test_gen,
        epochs=epochs, class_weights=class_weights, input_shape=(*img_size, 3)
    )
    all_metrics.append(efficientnet_metrics)

    # Compare all
    compare_all_models(all_metrics)
    logger.info("All transfer learning models trained and compared ✓")
