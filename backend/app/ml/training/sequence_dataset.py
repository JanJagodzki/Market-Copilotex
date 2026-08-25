from pathlib import Path

import numpy as np
import tensorflow as tf

from backend.app.ml.features.build_features import (
    FEATURE_COLUMNS,
)
from backend.app.ml.training.multi_horizon_dataset import (
    HORIZONS,
    get_split_ranges,
)


DATA_DIR = Path(
    "data/processed/sequences"
)

NUMBER_OF_FEATURES = len(
    FEATURE_COLUMNS
)


def get_files():
    return sorted(
        DATA_DIR.glob("*.npz")
    )


def get_range(
    horizon,
    split,
):
    if horizon not in HORIZONS:
        raise ValueError(
            f"Unknown horizon: {horizon}"
        )

    ranges = get_split_ranges(
        horizon
    )

    if split not in ranges:
        raise ValueError(
            f"Unknown split: {split}"
        )

    start_date, end_date = (
        ranges[split]
    )

    return (
        np.datetime64(
            str(start_date)
        ),
        np.datetime64(
            str(end_date)
        ),
    )


def calculate_train_stats(
    horizon,
):
    start_date, end_date = get_range(
        horizon=horizon,
        split="train",
    )

    total_count = 0

    total_sum = np.zeros(
        NUMBER_OF_FEATURES,
        dtype=np.float64,
    )

    total_square_sum = np.zeros(
        NUMBER_OF_FEATURES,
        dtype=np.float64,
    )

    files = get_files()

    print()
    print(
        f"Calculating scaler "
        f"for {horizon}D"
    )

    print(
        f"Train range: "
        f"{start_date} -> {end_date}"
    )

    for number, path in enumerate(
        files,
        start=1,
    ):
        with np.load(path) as data:
            x = data["x"].astype(
                np.float64
            )

            dates = data["dates"]

        mask = (
            (dates >= start_date)
            & (dates <= end_date)
        )

        values = x[mask]

        if len(values) == 0:
            continue

        total_count += len(
            values
        )

        total_sum += values.sum(
            axis=0
        )

        total_square_sum += (
            np.square(values)
            .sum(axis=0)
        )

        if number % 250 == 0:
            print(
                f"Scaler: "
                f"{number}/"
                f"{len(files)}"
            )

    if total_count == 0:
        raise RuntimeError(
            "No training rows "
            "for scaler"
        )

    mean = (
        total_sum
        / total_count
    )

    variance = (
        total_square_sum
        / total_count
        - np.square(mean)
    )

    variance = np.maximum(
        variance,
        1e-12,
    )

    std = np.sqrt(
        variance
    )

    std[
        std < 1e-8
    ] = 1.0

    print(
        f"Scaler rows: "
        f"{total_count:,}"
    )

    return (
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


def sequence_generator(
    split,
    horizon,
    mean,
    std,
    window_size=60,
    stride=1,
):
    if horizon not in HORIZONS:
        raise ValueError(
            f"Unknown horizon: {horizon}"
        )

    start_date, end_date = get_range(
        horizon=horizon,
        split=split,
    )

    label_key = (
        f"y_{horizon}d"
    )

    for path in get_files():
        with np.load(path) as data:
            x = data["x"]
            y = data[label_key]
            dates = data["dates"]

        if len(x) < window_size:
            continue

        for index in range(
            window_size - 1,
            len(x),
            stride,
        ):
            if y[index] < 0:
                continue

            origin_date = (
                dates[index]
            )

            if (
                origin_date < start_date
                or origin_date > end_date
            ):
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
                np.float32(
                    y[index]
                ),
            )


def count_sequences(
    split,
    horizon,
    window_size=60,
    stride=1,
):
    if horizon not in HORIZONS:
        raise ValueError(
            f"Unknown horizon: {horizon}"
        )

    start_date, end_date = get_range(
        horizon=horizon,
        split=split,
    )

    label_key = (
        f"y_{horizon}d"
    )

    total = 0

    for path in get_files():
        with np.load(path) as data:
            y = data[label_key]
            dates = data["dates"]

        if len(y) < window_size:
            continue

        indices = np.arange(
            window_size - 1,
            len(y),
            stride,
        )

        labels = y[
            indices
        ]

        origin_dates = dates[
            indices
        ]

        valid_labels = (
            labels >= 0
        )

        valid_dates = (
            (origin_dates >= start_date)
            & (origin_dates <= end_date)
        )

        total += int(
            np.sum(
                valid_labels
                & valid_dates
            )
        )

    return total


def make_dataset(
    split,
    horizon,
    mean,
    std,
    batch_size=64,
    window_size=60,
    stride=1,
):
    def generator():
        return sequence_generator(
            split=split,
            horizon=horizon,
            mean=mean,
            std=std,
            window_size=window_size,
            stride=stride,
        )

    dataset = (
        tf.data.Dataset
        .from_generator(
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
    )

    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=10000,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset
