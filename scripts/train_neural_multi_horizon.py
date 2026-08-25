import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from backend.app.ml.evaluation.classification import (
    classification_metrics,
)

from backend.app.ml.models.neural_models import (
    build_cnn,
    build_cnn_lstm,
    build_gru,
    build_lstm,
    build_mlp,
    build_patchtst,
    build_tcn,
    build_transformer,
)

from backend.app.ml.training.multi_horizon_dataset import (
    HORIZONS,
)

from backend.app.ml.training.sequence_dataset import (
    NUMBER_OF_FEATURES,
    calculate_train_stats,
    count_sequences,
    make_dataset,
)


MODEL_DIR = Path(
    "models/neural_multi_horizon"
)

WINDOW_SIZE = 60


MODEL_BUILDERS = {
    "MLP": build_mlp,
    "1D CNN": build_cnn,
    "LSTM": build_lstm,
    "GRU": build_gru,
    "TCN": build_tcn,
    "CNN-LSTM": build_cnn_lstm,
    "Transformer": build_transformer,
    "PatchTST": build_patchtst,
}


def configure_gpu():
    gpus = (
        tf.config
        .list_physical_devices(
            "GPU"
        )
    )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except RuntimeError:
            pass

    return gpus


def evaluate_model(
    model,
    dataset,
):
    probabilities = []
    labels = []

    for x_batch, y_batch in dataset:
        batch_probabilities = model(
            x_batch,
            training=False,
        ).numpy().reshape(-1)

        probabilities.append(
            batch_probabilities
        )

        labels.append(
            y_batch.numpy()
        )

    probabilities = np.concatenate(
        probabilities
    )

    y_true = np.concatenate(
        labels
    ).astype(int)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return classification_metrics(
        y_true,
        predictions,
        probabilities,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--train-stride",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--validation-stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=HORIZONS,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(
            MODEL_BUILDERS.keys()
        ),
    )

    args = parser.parse_args()

    for horizon in args.horizons:
        if horizon not in HORIZONS:
            raise ValueError(
                f"Invalid horizon: "
                f"{horizon}"
            )

    for model_name in args.models:
        if (
            model_name
            not in MODEL_BUILDERS
        ):
            raise ValueError(
                f"Invalid model: "
                f"{model_name}"
            )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gpus = configure_gpu()

    print()
    print(
        "MarketCopilotex "
        "Full Neural Benchmark"
    )

    print("=" * 70)

    print(
        f"TensorFlow: "
        f"{tf.__version__}"
    )

    print(
        f"GPU: {gpus}"
    )

    print(
        f"Horizons: "
        f"{args.horizons}"
    )

    print(
        f"Models: "
        f"{args.models}"
    )

    results = []

    input_shape = (
        WINDOW_SIZE,
        NUMBER_OF_FEATURES,
    )

    for horizon in args.horizons:
        print()
        print("#" * 70)
        print(
            f"HORIZON: "
            f"{horizon}D"
        )
        print("#" * 70)

        horizon_dir = (
            MODEL_DIR /
            f"{horizon}d"
        )

        horizon_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        scaler_path = (
            horizon_dir /
            "scaler.npz"
        )

        if scaler_path.exists():
            print(
                "Loading scaler"
            )

            with np.load(
                scaler_path
            ) as scaler:
                mean = scaler[
                    "mean"
                ]

                std = scaler[
                    "std"
                ]

        else:
            mean, std = (
                calculate_train_stats(
                    horizon
                )
            )

            np.savez(
                scaler_path,
                mean=mean,
                std=std,
            )

            print(
                f"Scaler saved: "
                f"{scaler_path}"
            )

        print()
        print(
            "Creating datasets..."
        )

        train_dataset = (
            make_dataset(
                split="train",
                horizon=horizon,
                mean=mean,
                std=std,
                batch_size=(
                    args.batch_size
                ),
                window_size=(
                    WINDOW_SIZE
                ),
                stride=(
                    args.train_stride
                ),
            )
        )

        validation_dataset = (
            make_dataset(
                split="validation",
                horizon=horizon,
                mean=mean,
                std=std,
                batch_size=(
                    args.batch_size
                ),
                window_size=(
                    WINDOW_SIZE
                ),
                stride=(
                    args.validation_stride
                ),
            )
        )

        full_validation_dataset = (
            make_dataset(
                split="validation",
                horizon=horizon,
                mean=mean,
                std=std,
                batch_size=(
                    args.batch_size
                ),
                window_size=(
                    WINDOW_SIZE
                ),
                stride=1,
            )
        )

        test_dataset = make_dataset(
            split="test",
            horizon=horizon,
            mean=mean,
            std=std,
            batch_size=(
                args.batch_size
            ),
            window_size=(
                WINDOW_SIZE
            ),
            stride=1,
        )

        train_samples = (
            count_sequences(
                split="train",
                horizon=horizon,
                window_size=(
                    WINDOW_SIZE
                ),
                stride=(
                    args.train_stride
                ),
            )
        )

        validation_samples = (
            count_sequences(
                split="validation",
                horizon=horizon,
                window_size=(
                    WINDOW_SIZE
                ),
                stride=(
                    args.validation_stride
                ),
            )
        )

        train_steps = (
            train_samples
            + args.batch_size
            - 1
        ) // args.batch_size

        validation_steps = (
            validation_samples
            + args.batch_size
            - 1
        ) // args.batch_size

        print(
            f"Train sequences: "
            f"{train_samples:,}"
        )

        print(
            f"Validation sequences: "
            f"{validation_samples:,}"
        )

        print(
            f"Train steps: "
            f"{train_steps:,}"
        )

        print(
            f"Validation steps: "
            f"{validation_steps:,}"
        )

        for model_name in (
            args.models
        ):
            print()
            print("=" * 70)

            print(
                f"{horizon}D - "
                f"{model_name}"
            )

            print("=" * 70)

            tf.keras.backend.clear_session()

            builder = (
                MODEL_BUILDERS[
                    model_name
                ]
            )

            model = builder(
                input_shape
            )

            callbacks = [
                tf.keras.callbacks
                .EarlyStopping(
                    monitor="val_auc",
                    mode="max",
                    patience=2,
                    restore_best_weights=True,
                ),

                tf.keras.callbacks
                .ReduceLROnPlateau(
                    monitor="val_auc",
                    mode="max",
                    factor=0.5,
                    patience=1,
                    min_lr=0.00001,
                ),
            ]

            repeated_train = (
                train_dataset.repeat()
            )

            repeated_validation = (
                validation_dataset
                .repeat()
            )

            start = time.time()

            model.fit(
                repeated_train,
                validation_data=(
                    repeated_validation
                ),
                epochs=args.epochs,
                steps_per_epoch=(
                    train_steps
                ),
                validation_steps=(
                    validation_steps
                ),
                callbacks=callbacks,
                verbose=1,
                shuffle=False,
            )

            seconds = (
                time.time()
                - start
            )

            print()
            print(
                "Full validation "
                "evaluation..."
            )

            validation_metrics = (
                evaluate_model(
                    model,
                    full_validation_dataset,
                )
            )

            print(
                "Full test "
                "evaluation..."
            )

            test_metrics = (
                evaluate_model(
                    model,
                    test_dataset,
                )
            )

            print()
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
                f"Test F1:        "
                f"{test_metrics['f1']:.4f}"
            )

            print(
                f"Test log loss:  "
                f"{test_metrics['log_loss']:.4f}"
            )

            print(
                f"Training time:  "
                f"{seconds:.1f}s"
            )

            filename = (
                model_name.lower()
                .replace(
                    " ",
                    "_",
                )
                .replace(
                    "-",
                    "_",
                )
            )

            model_path = (
                horizon_dir /
                f"{filename}.keras"
            )

            model.save(
                model_path
            )

            results.append(
                {
                    "horizon":
                        horizon,

                    "model":
                        model_name,

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

                    "seconds":
                        seconds,
                }
            )

    leaderboard = pd.DataFrame(
        results
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    path = (
        MODEL_DIR /
        f"leaderboard_"
        f"{timestamp}.csv"
    )

    leaderboard.to_csv(
        path,
        index=False,
    )

    validation_table = (
        leaderboard.pivot(
            index="model",
            columns="horizon",
            values="validation_auc",
        )
    )

    test_table = (
        leaderboard.pivot(
            index="model",
            columns="horizon",
            values="test_auc",
        )
    )

    print()
    print("=" * 90)
    print(
        "NEURAL VALIDATION "
        "ROC-AUC BY HORIZON"
    )
    print("=" * 90)

    print(
        validation_table.round(4)
    )

    print()
    print("=" * 90)
    print(
        "NEURAL TEST "
        "ROC-AUC BY HORIZON"
    )
    print("=" * 90)

    print(
        test_table.round(4)
    )

    print()
    print(
        f"Leaderboard saved: "
        f"{path}"
    )


if __name__ == "__main__":
    main()
