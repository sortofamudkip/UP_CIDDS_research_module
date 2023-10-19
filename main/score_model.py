from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from pandas import DataFrame


def mode_collapse_binary_stats(title):
    return DataFrame(
        {
            "accuracy": [0],
            "precision": [0],
            "recall": [0],
            "f1_score": [0],
            "confusion_matrix": ["only one class"],
        },
        index=[title],
    )


def f1_stats_one_epoch(y_true, y_pred, num_classes: int, pos_label="attacker"):
    if num_classes == 2:
        return f1_score(y_true, y_pred, pos_label=pos_label)
    elif num_classes == 5:
        return f1_score(y_true, y_pred, average="weighted")


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


def roc_auc_one_epoch(y_true, y_pred):
    """
    Computes the ROC AUC score for one epoch of a model's predictions.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: ROC AUC score.
    """
    return roc_auc_score(y_true, y_pred)


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
