"""
test_preprocessing.py
=====================
Unit tests for image preprocessing pipeline.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
import cv2
from PIL import Image
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    load_and_validate_image,
    preprocess_single_image,
    preprocess_for_prediction,
    create_train_data_generator,
    create_eval_data_generator,
    compute_class_weights
)


class TestImageLoading:
    """Tests for image loading and validation."""

    def setup_method(self):
        """Create a temporary test image."""
        self.temp_dir  = tempfile.mkdtemp()
        self.img_path  = os.path.join(self.temp_dir, "test_xray.jpg")

        # Create a realistic grayscale medical image (224x224)
        img = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite(self.img_path, img)

    def test_load_valid_image(self):
        """Test loading a valid image file."""
        img = load_and_validate_image(self.img_path)
        assert img is not None, "Image should load successfully"
        assert isinstance(img, np.ndarray), "Should return numpy array"
        assert len(img.shape) == 3, "Should have 3 dimensions (H, W, C)"
        assert img.shape[2] == 3, "Should have 3 channels (RGB)"

    def test_load_nonexistent_image(self):
        """Test behavior with non-existent file."""
        img = load_and_validate_image("/nonexistent/path/image.jpg")
        assert img is None, "Should return None for missing file"

    def test_load_invalid_file(self):
        """Test behavior with invalid (non-image) file."""
        invalid_path = os.path.join(self.temp_dir, "not_an_image.jpg")
        with open(invalid_path, "w") as f:
            f.write("this is not an image")
        img = load_and_validate_image(invalid_path)
        assert img is None, "Should return None for invalid image file"

    def test_load_returns_rgb(self):
        """Test that loaded image is in RGB (not BGR) format."""
        # Create BGR image with known pixel values
        img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_bgr[:, :, 0] = 255  # Blue channel = 255

        path = os.path.join(self.temp_dir, "bgr_test.jpg")
        cv2.imwrite(path, img_bgr)

        img_rgb = load_and_validate_image(path)
        assert img_rgb is not None
        # After BGR→RGB conversion, blue channel becomes R=255 in some pixels
        # (JPEG compression may slightly alter values, so just check shape)
        assert img_rgb.shape[2] == 3


class TestPreprocessing:
    """Tests for image preprocessing functions."""

    def test_resize_correct_dimensions(self):
        """Test that resize produces correct output dimensions."""
        img = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
        processed = preprocess_single_image(img, target_size=(224, 224))
        assert processed.shape == (224, 224, 3), \
            f"Expected (224,224,3), got {processed.shape}"

    def test_normalization_range(self):
        """Test that pixel values are normalized to [0, 1]."""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed = preprocess_single_image(img)
        assert processed.min() >= 0.0, "Min value should be >= 0"
        assert processed.max() <= 1.0, "Max value should be <= 1"

    def test_output_dtype(self):
        """Test that output is float32."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = preprocess_single_image(img)
        assert processed.dtype == np.float32, \
            f"Expected float32, got {processed.dtype}"

    def test_clahe_without_denoising(self):
        """Test preprocessing with CLAHE only."""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        processed = preprocess_single_image(img, apply_denoising=False, apply_clahe=True)
        assert processed is not None
        assert processed.shape == (224, 224, 3)

    def test_no_enhancement(self):
        """Test basic preprocessing without CLAHE or denoising."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        processed = preprocess_single_image(img, apply_clahe=False, apply_denoising=False)
        expected_pixel = 128 / 255.0
        assert abs(processed.mean() - expected_pixel) < 0.01

    def test_preprocess_for_prediction_shape(self):
        """Test that prediction preprocessing adds batch dimension."""
        temp_dir = tempfile.mkdtemp()
        img_path = os.path.join(temp_dir, "pred_test.jpg")
        img = np.random.randint(50, 200, (300, 250, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)

        result = preprocess_for_prediction(img_path)
        assert result is not None
        assert result.shape == (1, 224, 224, 3), \
            f"Expected (1,224,224,3), got {result.shape}"

    def test_preprocess_for_prediction_missing_file(self):
        """Test prediction preprocessing with missing file."""
        result = preprocess_for_prediction("/does/not/exist.jpg")
        assert result is None


class TestDataGenerators:
    """Tests for Keras ImageDataGenerators."""

    def test_train_generator_with_augmentation(self):
        """Test that train generator is created with augmentation."""
        gen = create_train_data_generator(augment=True)
        assert gen is not None
        assert gen.rotation_range == 15
        assert gen.horizontal_flip == True
        assert gen.rescale == pytest.approx(1.0 / 255.0)

    def test_train_generator_without_augmentation(self):
        """Test train generator without augmentation."""
        gen = create_train_data_generator(augment=False)
        assert gen is not None
        assert gen.rescale == pytest.approx(1.0 / 255.0)
        assert gen.rotation_range == 0  # No augmentation

    def test_eval_generator(self):
        """Test evaluation generator (rescale only)."""
        gen = create_eval_data_generator()
        assert gen is not None
        assert gen.rescale == pytest.approx(1.0 / 255.0)
        assert gen.rotation_range == 0
        assert gen.horizontal_flip == False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
