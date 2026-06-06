from src.database import register_model_version

metrics = {
    "accuracy": 0.9287,
    "precision": 0.9341,
    "recall": 0.9487,
    "f1_score": 0.9413,
    "roc_auc": 0.9713
}

register_model_version(
    model_name="DenseNet121_v1",
    architecture="densenet121",
    metrics=metrics,
    file_path="models/saved_models/densenet121.h5"
)

print("Model registered successfully!")