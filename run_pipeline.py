"""
run_pipeline.py
===============
Master end-to-end pipeline script for the Medical AI Platform.

Run this after downloading the Kaggle dataset to:
  1. Initialize the SQLite database
  2. Seed demo patient/prediction data
  3. Train the custom CNN model
  4. Train all transfer learning models (ResNet50, DenseNet121, EfficientNetB0)
  5. Compare model performance and save a report
  6. Register the best model in the database
  7. Log the full experiment to MLflow

Usage:
    conda activate medical_ai
    python src/run_pipeline.py --mode full
    python src/run_pipeline.py --mode db-only
    python src/run_pipeline.py --mode cnn-only
    python src/run_pipeline.py --mode transfer-only
    python src/run_pipeline.py --mode compare-only
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from src.cnn_model import MedicalCNN, TrainingPipeline

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from logger import get_logger

logger = get_logger("run_pipeline")


# ─────────────────────────────────────────────
# STEP FUNCTIONS
# ─────────────────────────────────────────────

def step_database():
    """Initialize database and seed demo data."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATABASE INITIALIZATION")
    logger.info("=" * 60)
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        db.initialize_database()
        logger.info("✅ Database tables created.")
        db.seed_sample_data(n_patients=200, n_predictions=6)
        logger.info("✅ Demo data seeded (200 patients, 6-month history).")
        return True
    except Exception as e:
        logger.error(f"❌ Database step failed: {e}")
        logger.debug(traceback.format_exc())
        return False


def step_train_cnn():
    """Train the custom CNN model."""
    logger.info("=" * 60)
    logger.info("STEP 2: TRAINING CUSTOM CNN")
    logger.info("=" * 60)
    try:
        from cnn_model import MedicalCNN, TrainingPipeline
        import yaml
        with open(PROJECT_ROOT / "config" / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        train_dir = PROJECT_ROOT / cfg["data"]["train_dir"]
        val_dir   = PROJECT_ROOT / cfg["data"]["val_dir"]
        test_dir  = PROJECT_ROOT / cfg["data"]["test_dir"]

        if not train_dir.exists():
            logger.error(f"❌ Training data not found at {train_dir}")
            logger.error("   Please download the dataset first:")
            logger.error("   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip")
            return False

        pipeline = TrainingPipeline(config_path=str(PROJECT_ROOT / "config" / "config.yaml"))
        history, model = pipeline.run(
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            test_dir=str(test_dir),
            save_path=str(PROJECT_ROOT / "models" / "saved_models" / "cnn_model.h5"),
        )
        logger.info("✅ Custom CNN training complete.")
        return True
    except Exception as e:
        logger.error(f"❌ CNN training failed: {e}")
        logger.debug(traceback.format_exc())
        return False


def step_transfer_learning():
    """Train all transfer learning models."""
    logger.info("=" * 60)
    logger.info("STEP 3: TRANSFER LEARNING (ResNet50, DenseNet121, EfficientNetB0)")
    logger.info("=" * 60)
    try:
        from transfer_learning import TransferLearningPipeline, ModelComparator
        import yaml
        with open(PROJECT_ROOT / "config" / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        train_dir = PROJECT_ROOT / cfg["data"]["train_dir"]
        if not train_dir.exists():
            logger.error(f"❌ Training data not found at {train_dir}")
            return False

        pipeline = TransferLearningPipeline(config_path=str(PROJECT_ROOT / "config" / "config.yaml"))

        for arch in ["resnet50", "densenet121", "efficientnetb0"]:
            logger.info(f"  → Training {arch.upper()} ...")
            try:
                pipeline.train_model(
                    architecture=arch,
                    train_dir=str(PROJECT_ROOT / cfg["data"]["train_dir"]),
                    val_dir=str(PROJECT_ROOT / cfg["data"]["val_dir"]),
                    test_dir=str(PROJECT_ROOT / cfg["data"]["test_dir"]),
                    save_path=str(PROJECT_ROOT / "models" / "saved_models" / f"{arch}_model.h5"),
                )
                logger.info(f"  ✅ {arch.upper()} training complete.")
            except Exception as inner_e:
                logger.warning(f"  ⚠️  {arch.upper()} failed: {inner_e}")
                continue

        return True
    except Exception as e:
        logger.error(f"❌ Transfer learning failed: {e}")
        logger.debug(traceback.format_exc())
        return False


def step_compare_models():
    """Compare all trained models and save report."""
    logger.info("=" * 60)
    logger.info("STEP 4: MODEL COMPARISON")
    logger.info("=" * 60)
    try:
        from transfer_learning import ModelComparator
        import yaml
        with open(PROJECT_ROOT / "config" / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        saved_dir = PROJECT_ROOT / "models" / "saved_models"
        model_files = list(saved_dir.glob("*.h5"))

        if not model_files:
            logger.warning("⚠️  No saved models found. Skipping comparison.")
            return True

        comparator = ModelComparator()
        test_dir = str(PROJECT_ROOT / cfg["data"]["test_dir"])
        comparator.compare_all(
            model_paths=[str(f) for f in model_files],
            test_dir=test_dir,
            output_path=str(PROJECT_ROOT / "models" / "comparison_report.csv"),
        )
        logger.info("✅ Model comparison report saved to models/comparison_report.csv")
        return True
    except Exception as e:
        logger.error(f"❌ Model comparison failed: {e}")
        logger.debug(traceback.format_exc())
        return False


def step_register_best_model():
    """Register the best model in the database."""
    logger.info("=" * 60)
    logger.info("STEP 5: REGISTERING BEST MODEL IN DATABASE")
    logger.info("=" * 60)
    try:
        import pandas as pd
        from database import DatabaseManager
        report_path = PROJECT_ROOT / "models" / "comparison_report.csv"
        if not report_path.exists():
            logger.warning("⚠️  Comparison report not found. Skipping registration.")
            return True

        df = pd.read_csv(report_path)
        best_row = df.loc[df["recall"].idxmax()]
        best_name = str(best_row.get("model_name", "best_model"))

        db = DatabaseManager()
        db.register_model_version(
            model_name=best_name,
            version="1.0.0",
            architecture=best_name,
            accuracy=float(best_row.get("accuracy", 0.0)),
            precision=float(best_row.get("precision", 0.0)),
            recall=float(best_row.get("recall", 0.0)),
            f1_score=float(best_row.get("f1_score", 0.0)),
            roc_auc=float(best_row.get("roc_auc", 0.0)),
            model_path=str(PROJECT_ROOT / "models" / "saved_models" / f"{best_name}.h5"),
            notes="Auto-registered by run_pipeline.py — best recall score.",
        )
        logger.info(f"✅ Best model '{best_name}' registered in database.")
        return True
    except Exception as e:
        logger.error(f"❌ Model registration failed: {e}")
        logger.debug(traceback.format_exc())
        return False


def step_mlflow_summary():
    """Print MLflow instructions."""
    logger.info("=" * 60)
    logger.info("STEP 6: MLFLOW TRACKING")
    logger.info("=" * 60)
    logger.info("  To view experiment results in the MLflow UI:")
    logger.info("  $ mlflow ui --host 0.0.0.0 --port 5000")
    logger.info("  Then open: http://localhost:5000")
    logger.info("✅ MLflow step complete.")
    return True


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

STEPS = {
    "db":       ("Database Init",          step_database),
    "cnn":      ("Custom CNN Training",    step_train_cnn),
    "transfer": ("Transfer Learning",      step_transfer_learning),
    "compare":  ("Model Comparison",       step_compare_models),
    "register": ("Best Model Registration",step_register_best_model),
    "mlflow":   ("MLflow Summary",         step_mlflow_summary),
}

MODE_STEPS = {
    "full":          ["db", "cnn", "transfer", "compare", "register", "mlflow"],
    "db-only":       ["db"],
    "cnn-only":      ["cnn"],
    "transfer-only": ["transfer"],
    "compare-only":  ["compare", "register"],
    "no-train":      ["db", "compare", "register", "mlflow"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Medical AI Platform — End-to-End Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python src/run_pipeline.py --mode full           # Run everything
  python src/run_pipeline.py --mode db-only        # Init DB only
  python src/run_pipeline.py --mode cnn-only       # Train CNN only
  python src/run_pipeline.py --mode transfer-only  # Transfer learning only
  python src/run_pipeline.py --mode compare-only   # Compare + register best
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=list(MODE_STEPS.keys()),
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue pipeline even if a step fails",
    )
    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║     MEDICAL AI PLATFORM — PIPELINE RUNNER            ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Project root: {PROJECT_ROOT}")

    steps_to_run = MODE_STEPS[args.mode]
    results = {}
    start_total = time.time()

    for step_key in steps_to_run:
        step_name, step_fn = STEPS[step_key]
        logger.info(f"\n▶ Running: {step_name}")
        t0 = time.time()
        success = step_fn()
        elapsed = round(time.time() - t0, 1)
        results[step_name] = ("✅ PASSED" if success else "❌ FAILED")
        logger.info(f"  Completed in {elapsed}s — {results[step_name]}")

        if not success and not args.skip_errors:
            logger.error("Pipeline halted due to error. Use --skip-errors to continue.")
            break

    total_elapsed = round(time.time() - start_total, 1)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for step_name, status in results.items():
        logger.info(f"  {status}  {step_name}")
    logger.info(f"\nTotal time: {total_elapsed}s")
    logger.info("=" * 60)

    all_passed = all("PASSED" in v for v in results.values())
    if all_passed:
        logger.info("\n🎉 All steps completed successfully!")
        logger.info("Next step → Run the Streamlit app:")
        logger.info("  streamlit run app/streamlit_app.py")
    else:
        logger.warning("\n⚠️  Some steps failed. Check logs above for details.")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
