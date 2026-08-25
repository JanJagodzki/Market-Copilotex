import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from xgboost import XGBClassifier

from backend.app.ml.evaluation.classification import (
    classification_metrics,
)

from backend.app.ml.training.multi_horizon_dataset import (
    HORIZONS,
    get_split_ranges,
    load_dataset,
    split_xy,
)


MODEL_DIR = Path(
    "models/multi_horizon"
)


def get_models():
    return {
        "Logistic Regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=200,
                        random_state=42,
                    ),
                ),
            ]
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        ),

        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=7,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
            thread_count=-1,
        ),
    }


def evaluate_model(
    model,
    x,
    y,
):
    prediction = model.predict(
        x
    )

    probability = (
        model.predict_proba(x)[:, 1]
    )

    return classification_metrics(
        y,
        prediction,
        probability,
    )


def evaluate_baseline(
    positive_rate,
    y,
):
    probability = np.full(
        len(y),
        positive_rate,
        dtype=np.float32,
    )

    prediction = (
        probability >= 0.5
    ).astype(int)

    return classification_metrics(
        y,
        prediction,
        probability,
    )


def train_horizon(
    horizon,
    train_sample,
):
    print()
    print("#" * 80)
    print(
        f"HORIZON: {horizon}D"
    )
    print("#" * 80)

    ranges = get_split_ranges(
        horizon
    )

    print()
    print("Date ranges:")

    for name, date_range in (
        ranges.items()
    ):
        print(
            f"{name:12} "
            f"{date_range[0]} "
            f"-> {date_range[1]}"
        )

    print()
    print("Loading data...")

    train = load_dataset(
        horizon=horizon,
        split="train",
        sample_percent=train_sample,
    )

    validation = load_dataset(
        horizon=horizon,
        split="validation",
        sample_percent=100,
    )

    test = load_dataset(
        horizon=horizon,
        split="test",
        sample_percent=100,
    )

    x_train, y_train = split_xy(
        train
    )

    x_validation, y_validation = (
        split_xy(validation)
    )

    x_test, y_test = split_xy(
        test
    )

    print()
    print(
        f"Train rows:      "
        f"{len(x_train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(x_validation):,}"
    )

    print(
        f"Test rows:       "
        f"{len(x_test):,}"
    )

    print()
    print(
        f"Train UP rate: "
        f"{y_train.mean():.4f}"
    )

    print(
        f"Validation UP rate: "
        f"{y_validation.mean():.4f}"
    )

    print(
        f"Test UP rate: "
        f"{y_test.mean():.4f}"
    )

    if (
        len(x_train) == 0
        or len(x_validation) == 0
        or len(x_test) == 0
    ):
        raise RuntimeError(
            f"Empty dataset for {horizon}D"
        )

    horizon_dir = (
        MODEL_DIR
        / f"{horizon}d"
    )

    horizon_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    positive_rate = float(
        y_train.mean()
    )

    validation_baseline = (
        evaluate_baseline(
            positive_rate,
            y_validation,
        )
    )

    test_baseline = (
        evaluate_baseline(
            positive_rate,
            y_test,
        )
    )

    results.append(
        {
            "horizon": horizon,
            "model": "Baseline",

            "validation_accuracy":
                validation_baseline[
                    "accuracy"
                ],

            "validation_balanced_accuracy":
                validation_baseline[
                    "balanced_accuracy"
                ],

            "validation_auc":
                validation_baseline[
                    "roc_auc"
                ],

            "test_accuracy":
                test_baseline[
                    "accuracy"
                ],

            "test_balanced_accuracy":
                test_baseline[
                    "balanced_accuracy"
                ],

            "test_f1":
                test_baseline[
                    "f1"
                ],

            "test_auc":
                test_baseline[
                    "roc_auc"
                ],

            "test_log_loss":
                test_baseline[
                    "log_loss"
                ],

            "seconds": 0,
        }
    )

    for name, model in (
        get_models().items()
    ):
        print()
        print("=" * 60)
        print(
            f"{horizon}D - {name}"
        )
        print("=" * 60)

        start = time.time()

        model.fit(
            x_train,
            y_train,
        )

        seconds = (
            time.time() - start
        )

        validation_metrics = (
            evaluate_model(
                model,
                x_validation,
                y_validation,
            )
        )

        test_metrics = (
            evaluate_model(
                model,
                x_test,
                y_test,
            )
        )

        print(
            f"Validation AUC: "
            f"{validation_metrics['roc_auc']:.4f}"
        )

        print(
            f"Test AUC:       "
            f"{test_metrics['roc_auc']:.4f}"
        )

        print(
            f"Test accuracy:  "
            f"{test_metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced acc:   "
            f"{test_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Training time:  "
            f"{seconds:.1f}s"
        )

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            horizon_dir / filename,
        )

        results.append(
            {
                "horizon": horizon,
                "model": name,

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_balanced_accuracy":
                    validation_metrics[
                        "balanced_accuracy"
                    ],

                "validation_auc":
                    validation_metrics[
                        "roc_auc"
                    ],

                "test_accuracy":
                    test_metrics[
                        "accuracy"
                    ],

                "test_balanced_accuracy":
                    test_metrics[
                        "balanced_accuracy"
                    ],

                "test_f1":
                    test_metrics[
                        "f1"
                    ],

                "test_auc":
                    test_metrics[
                        "roc_auc"
                    ],

                "test_log_loss":
                    test_metrics[
                        "log_loss"
                    ],

                "seconds": seconds,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=HORIZONS,
    )

    parser.add_argument(
        "--train-sample",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    for horizon in args.horizons:
        if horizon not in HORIZONS:
            raise ValueError(
                f"Invalid horizon: {horizon}"
            )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for horizon in args.horizons:
        results = train_horizon(
            horizon=horizon,
            train_sample=args.train_sample,
        )

        all_results.extend(
            results
        )

    leaderboard = pd.DataFrame(
        all_results
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        MODEL_DIR
        / f"leaderboard_{timestamp}.csv"
    )

    leaderboard.to_csv(
        path,
        index=False,
    )

    validation_auc = (
        leaderboard.pivot(
            index="model",
            columns="horizon",
            values="validation_auc",
        )
    )

    test_auc = (
        leaderboard.pivot(
            index="model",
            columns="horizon",
            values="test_auc",
        )
    )

    print()
    print()
    print("=" * 80)
    print(
        "VALIDATION ROC-AUC BY HORIZON"
    )
    print("=" * 80)

    print(
        validation_auc.round(4)
    )

    print()
    print("=" * 80)
    print(
        "TEST ROC-AUC BY HORIZON"
    )
    print("=" * 80)

    print(
        test_auc.round(4)
    )

    print()
    print(
        f"Leaderboard saved: {path}"
    )


if __name__ == "__main__":
    main()
