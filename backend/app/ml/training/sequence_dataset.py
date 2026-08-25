from pathlib import Path

import numpy as np
import tensorflow as tf

from backend.app.ml.features.build_features import FEATURE_COLUMNS


DATA_DIR = Path("data/processed/sequences")

NUMBER_OF_FEATURES = len(FEATURE_COLUMNS)


def get_files():
    return sorted(
        DATA_DIR.glob("*.npz")
    )


def calculate_train_stats(
    train_end="2024-12-31",
):
    train_end = np.datetime64(train_end)

    total = 0

    feature_sum = np.zeros(
        NUMBER_OF_FEATURES,
        dtype=np.float64,
    )

    feature_square_sum = np.zeros(
        NUMBER_OF_FEATURES,
        dtype=np.float64,
    )

    files = get_files()

    print(
        f"Calculating scaler from "
        f"{len(files)} files"
    )

    for number, path in enumerate(
        files,
        start=1,
    ):
        with np.load(path) as data:
            x = data["x"]
            dates = data["dates"]

        mask = dates <= train_end
        values = x[mask]

        if len(values) == 0:
            continue

        feature_sum += values.sum(
            axis=0,
            dtype=np.float64,
        )

        feature_square_sum += (
            np.square(
                values,
                dtype=np.float64,
            ).sum(axis=0)
        )

        total += len(values)

        if number % 250 == 0:
            print(
                f"Scaler: "
                f"{number}/{len(files)}"
            )

    if total == 0:
        raise RuntimeError(
            "No training data found"
        )

    mean = feature_sum / total

    variance = (
        feature_square_sum / total
        - np.square(mean)
    )

    variance = np.maximum(
        variance,
        0,
    )

    std = np.sqrt(variance)

    std[std < 1e-8] = 1.0

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def get_split_dates(split):
    if split == "train":
        return (
            np.datetime64("2000-01-01"),
            np.datetime64("2024-12-31"),
        )

    if split == "validation":
        return (
            np.datetime64("2025-01-01"),
            np.datetime64("2025-12-31"),
        )

    if split == "test":
        return (
            np.datetime64("2026-01-01"),
            np.datetime64("2026-12-31"),
        )

    raise ValueError(
        f"Unknown split: {split}"
    )


def sequence_generator(
    split,
    mean,
    std,
    window_size=60,
    stride=1,
):
    start_date, end_date = (
        get_split_dates(split)
    )

    files = get_files()

    for path in files:
        with np.load(path) as data:
            x = data["x"]
            y = data["y"]
            dates = data["dates"]

        if len(x) <= window_size:
            continue

        first_index = window_size - 1
        last_index = len(x) - 1

        for index in range(
            first_index,
            last_index,
            stride,
        ):
            if y[index] < 0:
                continue

            target_date = dates[
                index + 1
            ]

            if target_date < start_date:
                continue

            if target_date > end_date:
                continue

            window = x[
                index - window_size + 1:
                index + 1
            ]

            window = (
                window - mean
            ) / std

            yield (
                window.astype(
                    np.float32
                ),
                np.float32(y[index]),
            )

def count_sequences(
    split,
    window_size=60,
    stride=1,
):
    start_date, end_date = get_split_dates(split)

    total = 0

    for path in get_files():
        with np.load(path) as data:
            y = data["y"]
            dates = data["dates"]

        if len(y) <= window_size:
            continue

        indices = np.arange(
            window_size - 1,
            len(y) - 1,
            stride,
        )

        valid_labels = y[indices] >= 0

        target_dates = dates[
            indices + 1
        ]

        valid_dates = (
            (target_dates >= start_date)
            & (target_dates <= end_date)
        )

        total += np.sum(
            valid_labels & valid_dates
        )

    return int(total)
def count_sequences(
    split,
    window_size=60,
    stride=1,
):
    start_date, end_date = get_split_dates(split)

    total = 0

    for path in get_files():
        with np.load(path) as data:
            y = data["y"]
            dates = data["dates"]

        if len(y) <= window_size:
            continue

        indices = np.arange(
            window_size - 1,
            len(y) - 1,
            stride,
        )

        valid_labels = y[indices] >= 0

        target_dates = dates[
            indices + 1
        ]

        valid_dates = (
            (target_dates >= start_date)
            & (target_dates <= end_date)
        )

        total += np.sum(
            valid_labels & valid_dates
        )

    return int(total)
def make_dataset(
    split,
    mean,
    std,
    batch_size=256,
    window_size=60,
    stride=1,
):
    def generator():
        return sequence_generator(
            split=split,
            mean=mean,
            std=std,
            window_size=window_size,
            stride=stride,
        )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(
                    window_size,
                    NUMBER_OF_FEATURES,
                ),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.float32,
            ),
        ),
    )

    if split == "train":
        dataset = dataset.shuffle(
            10000,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset
