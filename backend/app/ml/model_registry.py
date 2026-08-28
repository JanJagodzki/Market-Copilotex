import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[3]
REGISTRY_PATH = Path(__file__).with_name(
    "active_models.json"
)


def load_model_registry():
    if not REGISTRY_PATH.exists():
        raise RuntimeError(
            "Active model registry does not exist"
        )

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = json.load(file)

    return registry


def get_model_path(model_info):
    return PROJECT_DIR / model_info["model_path"]


def get_scaler_path(model_info):
    return PROJECT_DIR / model_info["scaler_path"]


def get_model_quality(validation_auc):
    if validation_auc < 0.57:
        return "weak"

    if validation_auc < 0.62:
        return "limited"

    return "moderate"
