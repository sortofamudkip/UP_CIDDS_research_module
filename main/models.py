import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def y_train_test_split_summary(y, y_train, y_test, y_encoder):
    """Returns a dataframe with a summary of the train-test split for y.
    The data is assumed to already have been encoded (to int).
    Usage:
        y_encoder = LabelEncoder().fit(y)
        y = y_encoder.transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=55)
        y_train_test_split_summary(y, y_train, y_test, y_encoder)

    Args:
        y (np.array): original (all) y.
        y_train (np.array): y_train.
        y_test (np.array): y_test.
        y_encoder (sklearn.preprocessing.LabelEncoder): the encoder.
    """
    all_y_info = (
        pd.DataFrame(np.unique(y, return_counts=True)).T.set_index(0) / len(y) * 100
    )
    all_y_info_raw = pd.DataFrame(np.unique(y, return_counts=True)).T.set_index(0)
    y_train_info = (
        pd.DataFrame(np.unique(y_train, return_counts=True)).T.set_index(0)
        / len(y_train)
        * 100
    )
    y_train_info_raw = pd.DataFrame(np.unique(y_train, return_counts=True)).T.set_index(
        0
    )
    y_test_info = (
        pd.DataFrame(np.unique(y_test, return_counts=True)).T.set_index(0)
        / len(y_test)
        * 100
    )
    y_test_info_raw = pd.DataFrame(np.unique(y_test, return_counts=True)).T.set_index(0)
    y_train_test_stats = pd.concat(
        [
            all_y_info_raw,
            all_y_info,
            y_train_info_raw,
            y_train_info,
            y_test_info_raw,
            y_test_info,
        ],
        axis=1,
        ignore_index=True,
    ).rename(
        {
            0: "all y",
            1: "all y %",
            2: "y_train",
            3: "y_train %",
            4: "y_test",
            5: "y_test %",
        },
        axis=1,
    )
    return y_train_test_stats.rename(
        index={
            label: y_encoder.inverse_transform([label])[0]
            for label in range(0, len(np.unique(y)))
        }
    )


def knn_train_predict(
    X_train, X_test, y_train, y_test, y_encoder, n_neighbors=1
) -> np.array:
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model.predict(X_test)
