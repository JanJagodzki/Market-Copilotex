import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_DIR = Path(
    "models/neural_multi_horizon"
)

REGISTRY_PATH = Path(
    "backend/app/ml/active_models.json"
)


def model_filename(model_name):
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_latest_leaderboard():
    paths = sorted(
        MODEL_DIR.glob(
            "leaderboard_*.csv"
        )
    )

    if not paths:
        raise FileNotFoundError(
            "No neural leaderboard was found"
        )

    return paths[-1]


def select_best_models(data):
    required_columns = {
        "horizon",
        "model",
        "validation_auc",
        "test_auc",
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        names = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Leaderboard columns are missing: "
            f"{names}"
        )

    best_indices = (
        data.groupby("horizon")[
            "validation_auc"
        ]
        .idxmax()
    )

    return (
        data.loc[best_indices]
        .sort_values("horizon")
        .reset_index(drop=True)
    )


def build_registry(data, version):
    model_list = []

    for _, row in data.iterrows():
        horizon = int(row["horizon"])
        filename = model_filename(
            row["model"]
        )

        model_list.append(
            {
                "horizon": horizon,
                "model": row["model"],
                "validation_auc": round(
                    float(row["validation_auc"]),
                    6,
                ),
                "test_auc": round(
                    float(row["test_auc"]),
                    6,
                ),
                "model_path": (
                    "models/neural_multi_horizon/"
                    f"{horizon}d/{filename}.keras"
                ),
                "scaler_path": (
                    "models/neural_multi_horizon/"
                    f"{horizon}d/scaler.npz"
                ),
            }
        )

    return {
        "version": version,
        "selection_metric": "validation_auc",
        "window_size": 60,
        "models": model_list,
    }


def check_model_files(registry):
    missing = []

    for model_info in registry["models"]:
        for key in (
            "model_path",
            "scaler_path",
        ):
            path = Path(model_info[key])

            if not path.exists():
                missing.append(str(path))

    if missing:
        paths = "\n".join(missing)

        raise FileNotFoundError(
            "Selected model files are missing:\n"
            f"{paths}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    leaderboard_path = (
        args.leaderboard
        or find_latest_leaderboard()
    )

    data = pd.read_csv(
        leaderboard_path
    )

    best_models = select_best_models(
        data
    )

    version = (
        leaderboard_path.stem
        .replace("leaderboard_", "")
    )

    registry = build_registry(
        best_models,
        version,
    )

    check_model_files(registry)

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REGISTRY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            registry,
            file,
            indent=2,
        )

        file.write("\n")

    print(
        f"Registry saved: {REGISTRY_PATH}"
    )

    for model_info in registry["models"]:
        print(
            f"{model_info['horizon']}D: "
            f"{model_info['model']} "
            f"(validation AUC "
            f"{model_info['validation_auc']:.4f})"
        )


if __name__ == "__main__":
    main()
