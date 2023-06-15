from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
from pandas import DataFrame


def binary_stats(y_true, y_pred, title, y_encoder, pos_label="attacker"):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=pos_label, zero_division=0, average="binary"
    )
    stats = DataFrame(
        {
            "accuracy": [accuracy_score(y_true, y_pred)],
            "precision": [precision],
            "recall": [recall],
            "f1_score": [f1],
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


def multiclass_f1score_weighted(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted")
