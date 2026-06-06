"""
preprocessing.py
================
Complete image preprocessing pipeline for Medical AI Platform.

Handles:
  - Image loading & validation
  - Resizing & normalization
  - Denoising with OpenCV
  - Data augmentation
  - TensorFlow dataset preparation
  - Class imbalance handling
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import yaml
import warnings
warnings.filterwarnings("ignore")

from src.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path: str = "./config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration loaded from: {config_path}")
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Image Loading & Validation
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> Optional[np.ndarray]:
    """
    Load a single image, validate it, and return as numpy array.

    Args:
        image_path (str): Full path to the image file
        target_size (tuple): Target (width, height) for resizing

    Returns:
        np.ndarray or None: Loaded image array, or None if invalid
    """
    try:
        # Check file exists
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return None

        # Read with OpenCV (BGR format)
        img = cv2.imread(image_path)

        if img is None:
            logger.warning(f"Failed to read image: {image_path}")
            return None

        # Validate minimum size
        if img.shape[0] < 10 or img.shape[1] < 10:
            logger.warning(f"Image too small: {image_path}, shape={img.shape}")
            return None

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def validate_dataset_structure(dataset_path: str) -> Dict[str, int]:
    """
    Validate the chest_xray dataset folder structure.

    Args:
        dataset_path (str): Root path of the dataset

    Returns:
        dict: File count per split/class combination
    """
    logger.info("Validating dataset structure...")
    stats = {}

    required_splits = ["train", "test", "val"]
    required_classes = ["NORMAL", "PNEUMONIA"]

    for split in required_splits:
        for cls in required_classes:
            path = os.path.join(dataset_path, split, cls)
            if os.path.exists(path):
                count = len([
                    f for f in os.listdir(path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                stats[f"{split}/{cls}"] = count
                logger.info(f"  {split}/{cls}: {count} images")
            else:
                logger.error(f"  MISSING: {path}")
                stats[f"{split}/{cls}"] = 0

    total = sum(stats.values())
    logger.info(f"Total images found: {total}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing Functions
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_single_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    apply_denoising: bool = True,
    apply_clahe: bool = True
) -> np.ndarray:
    """
    Apply complete preprocessing pipeline to a single image.

    Pipeline:
      1. Resize to target dimensions
      2. Apply CLAHE for contrast enhancement (optional)
      3. Apply Gaussian denoising (optional)
      4. Normalize pixel values to [0, 1]

    Args:
        image (np.ndarray): Input RGB image
        target_size (tuple): Target (width, height)
        apply_denoising (bool): Whether to apply Gaussian blur denoising
        apply_clahe (bool): Whether to apply CLAHE contrast enhancement

    Returns:
        np.ndarray: Preprocessed image normalized to [0, 1]
    """
    # Step 1: Resize
    img = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Improves visibility of features in medical images
    if apply_clahe:
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    # Step 3: Denoising
    if apply_denoising:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 4: Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img


def preprocess_for_prediction(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> Optional[np.ndarray]:
    """
    Preprocess a single image for model prediction.
    Returns batch-ready array of shape (1, H, W, C).

    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target (height, width)

    Returns:
        np.ndarray: Batch-ready preprocessed image or None on failure
    """
    img = load_and_validate_image(image_path, target_size)
    if img is None:
        return None

    img = preprocess_single_image(img, target_size)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Data Generators
# ─────────────────────────────────────────────────────────────────────────────
def create_train_data_generator(
    augment: bool = True
) -> ImageDataGenerator:
    """
    Create ImageDataGenerator for training data with augmentation.

    Args:
        augment (bool): Whether to apply augmentation

    Returns:
        ImageDataGenerator: Configured generator
    """
    if augment:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=15,
            width_shift_range=0.10,
            height_shift_range=0.10,
            shear_range=0.10,
            zoom_range=0.10,
            horizontal_flip=True,
            fill_mode="nearest",
            brightness_range=[0.8, 1.2],
        )
        logger.info("Training generator created WITH augmentation")
    else:
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        logger.info("Training generator created WITHOUT augmentation")

    return datagen


def create_eval_data_generator() -> ImageDataGenerator:
    """
    Create ImageDataGenerator for validation/test data (no augmentation).

    Returns:
        ImageDataGenerator: Rescale-only generator
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    logger.info("Evaluation generator created (rescale only)")
    return datagen


def build_data_generators(
    dataset_path: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment_train: bool = True
) -> Tuple:
    """
    Build all three data generators: train, validation, test.

    Args:
        dataset_path (str): Root path of the dataset (contains train/val/test)
        image_size (tuple): (height, width) for resizing
        batch_size (int): Batch size for all generators
        augment_train (bool): Whether to augment training data

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    train_path = os.path.join(dataset_path, "train")
    val_path   = os.path.join(dataset_path, "val")
    test_path  = os.path.join(dataset_path, "test")

    train_datagen = create_train_data_generator(augment=augment_train)
    eval_datagen  = create_eval_data_generator()

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
        seed=42
    )

    val_generator = eval_datagen.flow_from_directory(
        val_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    test_generator = eval_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    logger.info(f"Train samples:      {train_generator.n}")
    logger.info(f"Validation samples: {val_generator.n}")
    logger.info(f"Test samples:       {test_generator.n}")
    logger.info(f"Class indices:      {train_generator.class_indices}")

    return train_generator, val_generator, test_generator


# ─────────────────────────────────────────────────────────────────────────────
# Class Weights (for imbalanced datasets)
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(train_generator) -> Dict[int, float]:
    """
    Compute class weights to handle class imbalance.

    Chest X-ray dataset typically has more PNEUMONIA than NORMAL images.
    Class weights penalize the model more for misclassifying the minority class.

    Args:
        train_generator: Keras ImageDataGenerator flow object

    Returns:
        dict: {class_index: weight} mapping
    """
    classes = train_generator.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes
    )

    weights_dict = {i: w for i, w in enumerate(class_weights)}
    logger.info(f"Class weights computed: {weights_dict}")
    logger.info("  Higher weight → model penalized more for that class misclassification")

    return weights_dict


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Statistics
# ─────────────────────────────────────────────────────────────────────────────
def get_dataset_statistics(dataset_path: str) -> pd.DataFrame:
    """
    Collect comprehensive dataset statistics for EDA.

    Args:
        dataset_path (str): Root dataset path

    Returns:
        pd.DataFrame: DataFrame with image metadata
    """
    logger.info("Collecting dataset statistics...")
    records = []

    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            folder = os.path.join(dataset_path, split, cls)
            if not os.path.exists(folder):
                continue

            for fname in os.listdir(folder):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath)

                if img is not None:
                    h, w, c = img.shape
                    fsize = os.path.getsize(fpath)
                    records.append({
                        "filename":  fname,
                        "split":     split,
                        "class":     cls,
                        "label":     1 if cls == "PNEUMONIA" else 0,
                        "height":    h,
                        "width":     w,
                        "channels":  c,
                        "file_size": fsize,
                        "aspect_ratio": round(w / h, 3),
                    })

    df = pd.DataFrame(records)
    logger.info(f"Dataset statistics collected: {len(df)} records")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pixel Intensity Analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_pixel_intensities(
    dataset_path: str,
    sample_size: int = 50
) -> Dict[str, np.ndarray]:
    """
    Sample images from each class and compute mean pixel intensities.

    Args:
        dataset_path (str): Root dataset path
        sample_size (int): Number of images to sample per class

    Returns:
        dict: {'NORMAL': mean_intensities, 'PNEUMONIA': mean_intensities}
    """
    logger.info("Analyzing pixel intensities...")
    results = {}

    for cls in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(dataset_path, "train", cls)
        files  = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:sample_size]

        intensities = []
        for fname in files:
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                intensities.extend(img.flatten().tolist())

        results[cls] = np.array(intensities)
        logger.info(f"  {cls}: mean={np.mean(results[cls]):.2f}, "
                    f"std={np.std(results[cls]):.2f}")

    return results


if __name__ == "__main__":
    """Quick test of preprocessing pipeline."""
    config = load_config()
    dataset_path = config["data"]["dataset_path"]

    # Validate structure
    stats = validate_dataset_structure(dataset_path)

    # Build generators
    train_gen, val_gen, test_gen = build_data_generators(
        dataset_path=dataset_path,
        image_size=tuple(config["image"]["target_size"]),
        batch_size=config["training"]["batch_size"]
    )

    # Compute class weights
    weights = compute_class_weights(train_gen)

    logger.info("Preprocessing pipeline test PASSED ✓")
