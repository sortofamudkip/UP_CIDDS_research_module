from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from pandas import DataFrame


def binary_stats(y_true, y_pred, title, y_encoder):
    pos_label = "attacker"
    stats = DataFrame(
        {
            "accuracy": [accuracy_score(y_true, y_pred)],
            "precision": [
                precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            ],
            "recall": [
                recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            ],
            "f1_score": [
                f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
            ],
            "confusion_matrix": [confusion_matrix(y_true, y_pred)],
        },
        index=[title],
    )
    return stats


def multiclass_stats(y_true, y_pred, title):
    stats = DataFrame(
        {
            "accuracy": [accuracy_score(y_true, y_pred)],
            "precision": [precision_score(y_true, y_pred, average="micro")],
            "recall": [recall_score(y_true, y_pred, average="micro")],
            "f1_score": [f1_score(y_true, y_pred, average="micro")],
            "confusion_matrix": [confusion_matrix(y_true, y_pred)],
        },
        index=[title],
    )
    return stats
