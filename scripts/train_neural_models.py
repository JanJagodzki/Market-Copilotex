import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
    build_tcn,
    build_transformer,
)
from backend.app.ml.training.sequence_dataset import (
    NUMBER_OF_FEATURES,
    calculate_train_stats,
    count_sequences,
    make_dataset,
)


MODEL_DIR = Path("models/neural")
WINDOW_SIZE = 60


MODEL_BUILDERS = {
    "MLP": build_mlp,
    "1D CNN": build_cnn,
    "LSTM": build_lstm,
    "GRU": build_gru,
    "TCN": build_tcn,
    "CNN-LSTM": build_cnn_lstm,
    "Transformer": build_transformer,
}


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except RuntimeError:
            pass

    return gpus


def evaluate_model(model, dataset):
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
        "--model",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gpus = configure_gpu()

    print()
    print("MarketCopilotex Neural Benchmark")
    print("=" * 60)

    print(
        f"TensorFlow: {tf.__version__}"
    )

    print(
        f"GPU: {gpus}"
    )

    print(
        f"Window size: {WINDOW_SIZE}"
    )

    print(
        f"Features: {NUMBER_OF_FEATURES}"
    )

    scaler_path = (
        MODEL_DIR / "scaler.npz"
    )

    if scaler_path.exists():
        print()
        print("Loading existing scaler")

        with np.load(scaler_path) as scaler:
            mean = scaler["mean"]
            std = scaler["std"]

    else:
        print()
        print(
            "Calculating scaler from TRAIN data"
        )

        mean, std = (
            calculate_train_stats()
        )

        np.savez(
            scaler_path,
            mean=mean,
            std=std,
        )

        print(
            f"Scaler saved: {scaler_path}"
        )

    print()
    print("Creating datasets...")

    train_dataset = make_dataset(
        split="train",
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        window_size=WINDOW_SIZE,
        stride=args.train_stride,
    )

    validation_dataset = make_dataset(
        split="validation",
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        window_size=WINDOW_SIZE,
        stride=args.validation_stride,
    )

    full_validation_dataset = make_dataset(
        split="validation",
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        window_size=WINDOW_SIZE,
        stride=1,
    )

    test_dataset = make_dataset(
        split="test",
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        window_size=WINDOW_SIZE,
        stride=1,
    )

    print()
    print("Counting sequences...")

    train_samples = count_sequences(
        split="train",
        window_size=WINDOW_SIZE,
        stride=args.train_stride,
    )

    validation_samples = count_sequences(
        split="validation",
        window_size=WINDOW_SIZE,
        stride=args.validation_stride,
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

    print()
    print(
        f"Train sequences:      "
        f"{train_samples:,}"
    )

    print(
        f"Validation sequences: "
        f"{validation_samples:,}"
    )

    print(
        f"Steps per epoch:      "
        f"{train_steps:,}"
    )

    print(
        f"Validation steps:     "
        f"{validation_steps:,}"
    )

    input_shape = (
        WINDOW_SIZE,
        NUMBER_OF_FEATURES,
    )

    if args.model:
        if args.model not in MODEL_BUILDERS:
            print()
            print(
                "Unknown model:",
                args.model,
            )

            print(
                "Available models:",
                ", ".join(
                    MODEL_BUILDERS.keys()
                ),
            )

            return

        models_to_train = {
            args.model:
            MODEL_BUILDERS[args.model]
        }

    else:
        models_to_train = (
            MODEL_BUILDERS
        )

    results = []

    for name, builder in (
        models_to_train.items()
    ):
        print()
        print("=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        tf.keras.backend.clear_session()

        model = builder(
            input_shape
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=2,
                restore_best_weights=True,
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                mode="max",
                factor=0.5,
                patience=1,
                min_lr=0.00001,
            ),
        ]

        repeated_train_dataset = (
            train_dataset.repeat()
        )

        repeated_validation_dataset = (
            validation_dataset.repeat()
        )

        start = time.time()

        model.fit(
            repeated_train_dataset,
            validation_data=(
                repeated_validation_dataset
            ),
            epochs=args.epochs,
            steps_per_epoch=train_steps,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
            shuffle=False,
        )

        training_time = (
            time.time() - start
        )

        print()
        print(
            "Evaluating full validation data..."
        )

        validation_metrics = (
            evaluate_model(
                model,
                full_validation_dataset,
            )
        )

        print(
            "Evaluating full test data..."
        )

        test_metrics = evaluate_model(
            model,
            test_dataset,
        )

        print()
        print(f"Model: {name}")

        print(
            f"Validation accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation AUC:      "
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
            f"Test AUC:            "
            f"{test_metrics['roc_auc']:.4f}"
        )

        print(
            f"Test log loss:       "
            f"{test_metrics['log_loss']:.4f}"
        )

        print(
            f"Training time:       "
            f"{training_time:.1f}s"
        )

        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        model_path = (
            MODEL_DIR
            / f"{filename}.keras"
        )

        model.save(
            model_path
        )

        print(
            f"Model saved: {model_path}"
        )

        results.append(
            {
                "model": name,

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_auc":
                    validation_metrics[
                        "roc_auc"
                    ],

                "test_accuracy":
                    test_metrics[
                        "accuracy"
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
                    training_time,
            }
        )

    leaderboard = pd.DataFrame(
        results
    )

    leaderboard = (
        leaderboard.sort_values(
            "validation_auc",
            ascending=False,
        )
    )

    print()
    print("=" * 80)
    print(
        "MARKETCOPILOTEX "
        "NEURAL MODEL LEADERBOARD"
    )
    print("=" * 80)

    print(
        leaderboard.to_string(
            index=False,
            formatters={
                "validation_accuracy":
                    "{:.4f}".format,

                "validation_auc":
                    "{:.4f}".format,

                "test_accuracy":
                    "{:.4f}".format,

                "test_f1":
                    "{:.4f}".format,

                "test_auc":
                    "{:.4f}".format,

                "test_log_loss":
                    "{:.4f}".format,

                "seconds":
                    "{:.1f}".format,
            },
        )
    )

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    leaderboard_path = (
        MODEL_DIR
        / f"leaderboard_{timestamp}.csv"
    )

    leaderboard.to_csv(
        leaderboard_path,
        index=False,
    )

    print()
    print(
        f"Leaderboard saved: "
        f"{leaderboard_path}"
    )


if __name__ == "__main__":
    main()