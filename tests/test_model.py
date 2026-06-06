"""
test_model.py
=============
Unit tests for CNN model architecture, training utilities, and evaluation functions.
Run with: pytest tests/test_model.py -v
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def dummy_images_batch():
    """Batch of 8 random images simulating preprocessed X-rays (224, 224, 3)."""
    np.random.seed(42)
    return np.random.rand(8, 224, 224, 3).astype(np.float32)


@pytest.fixture
def dummy_labels_binary():
    """Binary labels for the dummy batch (0=NORMAL, 1=PNEUMONIA)."""
    return np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.float32)


@pytest.fixture
def dummy_probabilities():
    """Simulated model output probabilities."""
    np.random.seed(0)
    return np.clip(np.random.rand(100), 0.01, 0.99).astype(np.float32)


@pytest.fixture
def dummy_true_labels():
    """Ground truth labels matching dummy_probabilities."""
    np.random.seed(1)
    return (np.random.rand(100) > 0.5).astype(np.int32)


# ─────────────────────────────────────────────
# CNN ARCHITECTURE TESTS
# ─────────────────────────────────────────────

class TestCNNArchitecture:
    """Tests for MedicalCNN model construction."""

    def test_model_builds_without_error(self):
        """MedicalCNN should instantiate and compile without raising."""
        try:
            import tensorflow as tf
            from cnn_model import MedicalCNN
            model_builder = MedicalCNN(input_shape=(224, 224, 3), num_classes=1)
            model = model_builder.build()
            assert model is not None
        except ImportError:
            pytest.skip("TensorFlow not available in this environment")

    def test_model_output_shape(self, dummy_images_batch):
        """Model should output (batch_size, 1) for binary classification."""
        try:
            import tensorflow as tf
            from cnn_model import MedicalCNN
            model_builder = MedicalCNN(input_shape=(224, 224, 3), num_classes=1)
            model = model_builder.build()
            output = model.predict(dummy_images_batch, verbose=0)
            assert output.shape == (8, 1), f"Expected (8,1), got {output.shape}"
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_model_output_range(self, dummy_images_batch):
        """Sigmoid outputs must be in [0, 1]."""
        try:
            import tensorflow as tf
            from cnn_model import MedicalCNN
            model_builder = MedicalCNN(input_shape=(224, 224, 3), num_classes=1)
            model = model_builder.build()
            output = model.predict(dummy_images_batch, verbose=0)
            assert np.all(output >= 0.0), "Output below 0"
            assert np.all(output <= 1.0), "Output above 1"
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_model_has_batch_normalization(self):
        """Model should include BatchNormalization layers."""
        try:
            import tensorflow as tf
            from cnn_model import MedicalCNN
            model_builder = MedicalCNN(input_shape=(224, 224, 3), num_classes=1)
            model = model_builder.build()
            layer_types = [type(layer).__name__ for layer in model.layers]
            assert "BatchNormalization" in layer_types, "No BatchNormalization layers found"
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_model_has_dropout(self):
        """Model should include Dropout for regularization."""
        try:
            import tensorflow as tf
            from cnn_model import MedicalCNN
            model_builder = MedicalCNN(input_shape=(224, 224, 3), num_classes=1)
            model = model_builder.build()
            layer_types = [type(layer).__name__ for layer in model.layers]
            assert "Dropout" in layer_types, "No Dropout layers found"
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_model_parameter_count_reasonable(self):
        """Model should have > 100k and < 50M parameters (sanity check)."""
        try:
            import tensorflow as tf
            from cnn_model import MedicalCNN
            model_builder = MedicalCNN(input_shape=(224, 224, 3), num_classes=1)
            model = model_builder.build()
            total_params = model.count_params()
            assert total_params > 100_000, f"Too few params: {total_params}"
            assert total_params < 50_000_000, f"Too many params: {total_params}"
        except ImportError:
            pytest.skip("TensorFlow not available")


# ─────────────────────────────────────────────
# METRICS CALCULATION TESTS
# ─────────────────────────────────────────────

class TestMetricsCalculation:
    """Tests for evaluation metric functions."""

    def test_accuracy_perfect(self):
        """Accuracy = 1.0 when all predictions are correct."""
        from sklearn.metrics import accuracy_score
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        acc = accuracy_score(y_true, y_pred)
        assert acc == 1.0

    def test_recall_critical_for_medical(self, dummy_probabilities, dummy_true_labels):
        """
        Recall (sensitivity) must be computed correctly.
        Medical context: FN (missed pneumonia) is more dangerous than FP.
        """
        from sklearn.metrics import recall_score
        y_pred = (dummy_probabilities >= 0.5).astype(int)
        recall = recall_score(dummy_true_labels, y_pred, zero_division=0)
        assert 0.0 <= recall <= 1.0, f"Recall out of range: {recall}"

    def test_roc_auc_random_is_near_05(self, dummy_probabilities, dummy_true_labels):
        """ROC-AUC for random predictions should be near 0.5."""
        from sklearn.metrics import roc_auc_score
        np.random.seed(999)
        random_probs = np.random.rand(len(dummy_true_labels))
        auc = roc_auc_score(dummy_true_labels, random_probs)
        assert 0.35 <= auc <= 0.65, f"Random AUC far from 0.5: {auc}"

    def test_perfect_classifier_metrics(self):
        """A perfect classifier should achieve accuracy=recall=precision=F1=1.0."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        assert accuracy_score(y_true, y_pred) == 1.0
        assert recall_score(y_true, y_pred)   == 1.0
        assert precision_score(y_true, y_pred) == 1.0
        assert f1_score(y_true, y_pred)        == 1.0

    def test_threshold_effect_on_recall(self):
        """Lowering threshold should increase recall (more positives predicted)."""
        from sklearn.metrics import recall_score
        np.random.seed(5)
        y_true = (np.random.rand(200) > 0.5).astype(int)
        probs  = np.random.rand(200)

        recall_high_thresh = recall_score(y_true, (probs >= 0.7).astype(int), zero_division=0)
        recall_low_thresh  = recall_score(y_true, (probs >= 0.3).astype(int), zero_division=0)
        assert recall_low_thresh >= recall_high_thresh, \
            "Lower threshold should not decrease recall"

    def test_confusion_matrix_values(self):
        """Verify confusion matrix extracts TP, TN, FP, FN correctly."""
        from sklearn.metrics import confusion_matrix
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        assert tp == 2  # Correctly predicted pneumonia
        assert tn == 2  # Correctly predicted normal
        assert fn == 1  # Missed pneumonia (dangerous!)
        assert fp == 1  # False alarm


# ─────────────────────────────────────────────
# TRANSFER LEARNING TESTS
# ─────────────────────────────────────────────

class TestTransferLearning:
    """Tests for transfer learning model construction."""

    @pytest.mark.parametrize("architecture", ["resnet50", "densenet121", "efficientnetb0"])
    def test_transfer_model_builds(self, architecture):
        """Each transfer learning architecture should build without error."""
        try:
            import tensorflow as tf
            from transfer_learning import TransferLearningPipeline
            pipeline = TransferLearningPipeline()
            model = pipeline.build_model(architecture=architecture, input_shape=(224, 224, 3))
            assert model is not None, f"{architecture} model is None"
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_base_layers_frozen_initially(self):
        """Base model layers should be frozen before fine-tuning."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50
            base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
            base.trainable = False
            for layer in base.layers:
                assert not layer.trainable, f"Layer {layer.name} should be frozen"
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_fine_tuning_unfreezes_layers(self):
        """After fine-tuning phase, top N layers should be trainable."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications import ResNet50
            base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
            base.trainable = False
            # Unfreeze last 20 layers (simulating fine-tuning)
            for layer in base.layers[-20:]:
                layer.trainable = True
            trainable_count = sum(1 for l in base.layers if l.trainable)
            assert trainable_count > 0, "No layers were unfrozen"
        except ImportError:
            pytest.skip("TensorFlow not available")


# ─────────────────────────────────────────────
# MODEL SAVE / LOAD TESTS
# ─────────────────────────────────────────────

class TestModelPersistence:
    """Tests for saving and loading models."""

    def test_model_save_and_load(self, tmp_path, dummy_images_batch):
        """Saved model should produce identical predictions when loaded."""
        try:
            import tensorflow as tf
            from cnn_model import MedicalCNN
            save_path = str(tmp_path / "test_model.h5")

            # Build and save
            builder = MedicalCNN(input_shape=(224, 224, 3), num_classes=1)
            model = builder.build()
            original_preds = model.predict(dummy_images_batch, verbose=0)
            model.save(save_path)

            # Load and predict
            loaded_model = tf.keras.models.load_model(save_path)
            loaded_preds = loaded_model.predict(dummy_images_batch, verbose=0)

            np.testing.assert_allclose(original_preds, loaded_preds, rtol=1e-5,
                                       err_msg="Loaded model predictions differ from original")
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_model_file_exists_after_save(self, tmp_path):
        """Model .h5 file should exist after saving."""
        try:
            import tensorflow as tf
            from cnn_model import MedicalCNN
            save_path = tmp_path / "test_save.h5"
            builder = MedicalCNN(input_shape=(224, 224, 3), num_classes=1)
            model = builder.build()
            model.save(str(save_path))
            assert save_path.exists(), "Model file was not created"
            assert save_path.stat().st_size > 0, "Model file is empty"
        except ImportError:
            pytest.skip("TensorFlow not available")
