import numpy as np

from backend.app.ml.features.build_features import (
    FEATURE_COLUMNS,
)
from backend.app.ml.model_registry import (
    get_model_path,
    get_model_quality,
    get_scaler_path,
    load_model_registry,
)


MODEL_CACHE = {}


class PredictionError(RuntimeError):
    pass


def find_missing_files(model_list):
    missing = []

    for model_info in model_list:
        model_path = get_model_path(
            model_info
        )
        scaler_path = get_scaler_path(
            model_info
        )

        if not model_path.exists():
            missing.append(str(model_path))

        if not scaler_path.exists():
            missing.append(str(scaler_path))

    return missing


def prepare_window(
    feature_rows,
    mean,
    std,
    window_size,
):
    if len(feature_rows) < window_size:
        raise PredictionError(
            f"At least {window_size} feature rows are required"
        )

    selected_rows = feature_rows[-window_size:]

    values = np.array(
        [
            [
                getattr(row, column)
                for column in FEATURE_COLUMNS
            ]
            for row in selected_rows
        ],
        dtype=np.float32,
    )

    if not np.isfinite(values).all():
        raise PredictionError(
            "Latest feature window contains missing values"
        )

    if (
        mean.shape[0] != values.shape[1]
        or std.shape[0] != values.shape[1]
    ):
        raise PredictionError(
            "Scaler does not match the feature list"
        )

    safe_std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    normalized = (
        values - mean
    ) / safe_std

    return np.expand_dims(
        normalized.astype(np.float32),
        axis=0,
    )


def load_neural_model(path):
    cache_key = str(path)

    if cache_key not in MODEL_CACHE:
        import tensorflow as tf

        MODEL_CACHE[cache_key] = (
            tf.keras.models.load_model(
                path,
                compile=False,
            )
        )

    return MODEL_CACHE[cache_key]


def predict_for_rows(feature_rows):
    registry = load_model_registry()
    model_list = registry["models"]
    window_size = registry["window_size"]

    missing_files = find_missing_files(
        model_list
    )

    if missing_files:
        joined_paths = ", ".join(
            missing_files
        )

        raise PredictionError(
            "Model files are missing: "
            f"{joined_paths}"
        )

    predictions = []

    for model_info in model_list:
        model_path = get_model_path(
            model_info
        )
        scaler_path = get_scaler_path(
            model_info
        )

        with np.load(scaler_path) as scaler:
            mean = scaler["mean"]
            std = scaler["std"]

        window = prepare_window(
            feature_rows=feature_rows,
            mean=mean,
            std=std,
            window_size=window_size,
        )

        model = load_neural_model(
            model_path
        )

        output = model(
            window,
            training=False,
        )

        probability_up = float(
            np.asarray(output).reshape(-1)[0]
        )

        probability_up = min(
            max(probability_up, 0.0),
            1.0,
        )

        predictions.append(
            {
                "horizon_days": model_info[
                    "horizon"
                ],
                "model": model_info["model"],
                "probability_up": round(
                    probability_up,
                    6,
                ),
                "direction": (
                    "up"
                    if probability_up >= 0.5
                    else "down"
                ),
                "validation_auc": model_info[
                    "validation_auc"
                ],
                "test_auc": model_info[
                    "test_auc"
                ],
                "quality": get_model_quality(
                    model_info[
                        "validation_auc"
                    ]
                ),
            }
        )

    return predictions
