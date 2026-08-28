import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backend.app.ml.features.build_features import (
    FEATURE_COLUMNS,
)
from backend.app.ml.prediction_service import (
    PredictionError,
    prepare_window,
)
from scripts.promote_neural_models import (
    build_registry,
    select_best_models,
)


class PredictionTests(unittest.TestCase):
    def make_feature_rows(self, count=60):
        rows = []

        for number in range(count):
            values = {
                column: float(number + 1)
                for column in FEATURE_COLUMNS
            }

            rows.append(
                SimpleNamespace(**values)
            )

        return rows

    def test_prepare_window(self):
        rows = self.make_feature_rows()
        mean = np.zeros(
            len(FEATURE_COLUMNS),
            dtype=np.float32,
        )
        std = np.ones(
            len(FEATURE_COLUMNS),
            dtype=np.float32,
        )

        result = prepare_window(
            feature_rows=rows,
            mean=mean,
            std=std,
            window_size=60,
        )

        self.assertEqual(
            result.shape,
            (1, 60, len(FEATURE_COLUMNS)),
        )

    def test_prepare_window_needs_60_rows(self):
        rows = self.make_feature_rows(
            count=59
        )
        mean = np.zeros(
            len(FEATURE_COLUMNS),
            dtype=np.float32,
        )
        std = np.ones(
            len(FEATURE_COLUMNS),
            dtype=np.float32,
        )

        with self.assertRaises(
            PredictionError
        ):
            prepare_window(
                feature_rows=rows,
                mean=mean,
                std=std,
                window_size=60,
            )

    def test_best_model_uses_validation_auc(self):
        data = pd.DataFrame(
            [
                {
                    "horizon": 5,
                    "model": "LSTM",
                    "validation_auc": 0.55,
                    "test_auc": 0.70,
                },
                {
                    "horizon": 5,
                    "model": "GRU",
                    "validation_auc": 0.56,
                    "test_auc": 0.60,
                },
            ]
        )

        result = select_best_models(data)

        self.assertEqual(
            result.iloc[0]["model"],
            "GRU",
        )

    def test_registry_builds_model_path(self):
        data = pd.DataFrame(
            [
                {
                    "horizon": 20,
                    "model": "1D CNN",
                    "validation_auc": 0.57,
                    "test_auc": 0.58,
                }
            ]
        )

        registry = build_registry(
            data,
            version="test",
        )

        self.assertEqual(
            registry["models"][0][
                "model_path"
            ],
            (
                "models/neural_multi_horizon/"
                "20d/1d_cnn.keras"
            ),
        )


if __name__ == "__main__":
    unittest.main()
