import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.app.ml.evaluation.classification import (
    classification_metrics,
)
from backend.app.ml.training.dataset import (
    load_dataset,
    split_xy,
)


MODEL_DIR = Path("models/classical")


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
    prediction = model.predict(x)

    probability = model.predict_proba(x)[:, 1]

    return classification_metrics(
        y,
        prediction,
        probability,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-sample",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    print()
    print("Loading datasets...")
    print()

    train = load_dataset(
        start_date="2000-01-01",
        end_date="2024-12-31",
        sample_percent=args.train_sample,
    )

    validation = load_dataset(
        start_date="2025-01-01",
        end_date="2025-12-31",
        sample_percent=100,
    )

    test = load_dataset(
        start_date="2026-01-01",
        end_date="2026-12-31",
        sample_percent=100,
    )

    x_train, y_train = split_xy(train)
    x_validation, y_validation = split_xy(
        validation
    )
    x_test, y_test = split_xy(test)

    print(f"Train rows:      {len(x_train):,}")
    print(
        f"Validation rows: {len(x_validation):,}"
    )
    print(f"Test rows:       {len(x_test):,}")

    print()
    print(
        f"Train positive rate: "
        f"{y_train.mean():.4f}"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    baseline_probability = float(
        y_train.mean()
    )

    baseline_prediction = (
        pd.Series(
            baseline_probability,
            index=y_validation.index,
        )
        >= 0.5
    ).astype(int)

    baseline_validation = (
        classification_metrics(
            y_validation,
            baseline_prediction,
            pd.Series(
                baseline_probability,
                index=y_validation.index,
            ),
        )
    )

    baseline_test_prediction = (
        pd.Series(
            baseline_probability,
            index=y_test.index,
        )
        >= 0.5
    ).astype(int)

    baseline_test = classification_metrics(
        y_test,
        baseline_test_prediction,
        pd.Series(
            baseline_probability,
            index=y_test.index,
        ),
    )

    results.append(
        {
            "model": "Baseline",
            "validation_accuracy": (
                baseline_validation["accuracy"]
            ),
            "validation_auc": (
                baseline_validation["roc_auc"]
            ),
            "test_accuracy": (
                baseline_test["accuracy"]
            ),
            "test_f1": baseline_test["f1"],
            "test_auc": (
                baseline_test["roc_auc"]
            ),
            "test_log_loss": (
                baseline_test["log_loss"]
            ),
            "seconds": 0,
        }
    )

    models = get_models()

    for name, model in models.items():
        print()
        print("=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        start = time.time()

        model.fit(
            x_train,
            y_train,
        )

        seconds = time.time() - start

        validation_metrics = evaluate_model(
            model,
            x_validation,
            y_validation,
        )

        test_metrics = evaluate_model(
            model,
            x_test,
            y_test,
        )

        print(
            f"Validation accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation ROC-AUC:  "
            f"{validation_metrics['roc_auc']:.4f}"
        )

        print(
            f"Test accuracy:       "
            f"{test_metrics['accuracy']:.4f}"
        )

        print(
            f"Test F1:             "
            f"{test_metrics['f1']:.4f}"
        )

        print(
            f"Test ROC-AUC:        "
            f"{test_metrics['roc_auc']:.4f}"
        )

        print(
            f"Training time:       "
            f"{seconds:.1f}s"
        )

        filename = (
            name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            MODEL_DIR / filename,
        )

        results.append(
            {
                "model": name,
                "validation_accuracy": (
                    validation_metrics[
                        "accuracy"
                    ]
                ),
                "validation_auc": (
                    validation_metrics[
                        "roc_auc"
                    ]
                ),
                "test_accuracy": (
                    test_metrics["accuracy"]
                ),
                "test_f1": (
                    test_metrics["f1"]
                ),
                "test_auc": (
                    test_metrics["roc_auc"]
                ),
                "test_log_loss": (
                    test_metrics["log_loss"]
                ),
                "seconds": seconds,
            }
        )

    leaderboard = pd.DataFrame(
        results
    )

    leaderboard = leaderboard.sort_values(
        "validation_auc",
        ascending=False,
    )

    print()
    print()
    print("=" * 80)
    print("MARKETCOPILOTEX MODEL LEADERBOARD")
    print("=" * 80)

    print(
        leaderboard.to_string(
            index=False,
            formatters={
                "validation_accuracy": (
                    "{:.4f}".format
                ),
                "validation_auc": (
                    "{:.4f}".format
                ),
                "test_accuracy": (
                    "{:.4f}".format
                ),
                "test_f1": (
                    "{:.4f}".format
                ),
                "test_auc": (
                    "{:.4f}".format
                ),
                "test_log_loss": (
                    "{:.4f}".format
                ),
                "seconds": (
                    "{:.1f}".format
                ),
            },
        )
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    result_path = (
        MODEL_DIR /
        f"leaderboard_{timestamp}.csv"
    )

    leaderboard.to_csv(
        result_path,
        index=False,
    )

    print()
    print(
        f"Leaderboard saved: {result_path}"
    )


if __name__ == "__main__":
    main()
