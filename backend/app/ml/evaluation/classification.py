import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    prediction,
    probability,
):
    probability = np.clip(
        probability,
        0.000001,
        0.999999,
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            prediction,
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
        "log_loss": log_loss(
            y_true,
            probability,
            labels=[0, 1],
        ),
    }
