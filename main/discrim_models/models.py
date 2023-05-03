import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def y_train_test_split_summary(y, y_train, y_test, y_encoder):
    """Returns a dataframe with a summary of the train-test split for y.
    The data is assumed to already have been encoded (to int).
    Usage:
        y_encoder = LabelEncoder().fit(y)
        y = y_encoder.transform(y)
        X, y, y_encoder, x_labels, x_encoders = train_test_split(X, y, test_size=0.33, random_state=55)
        y_train_test_split_summary(y, y_train, y_test, y_encoder)

    Args:
        y (np.array): original (all) y.
        y_train (np.array): y_train.
        y_test (np.array): y_test.
        y_encoder (sklearn.preprocessing.LabelEncoder): the encoder.
    """
    decoded_y = y_encoder.inverse_transform(y)
    all_y_info = (
        pd.DataFrame(np.unique(decoded_y, return_counts=True)).T.set_index(0)
        / len(decoded_y)
        * 100
    )
    all_y_info_raw = pd.DataFrame(np.unique(decoded_y, return_counts=True)).T.set_index(
        0
    )

    decoded_y_train = y_encoder.inverse_transform(y_train)
    y_train_info = (
        pd.DataFrame(np.unique(decoded_y_train, return_counts=True)).T.set_index(0)
        / len(decoded_y_train)
        * 100
    )
    y_train_info_raw = pd.DataFrame(
        np.unique(decoded_y_train, return_counts=True)
    ).T.set_index(0)

    decoded_y_test = y_encoder.inverse_transform(y_test)
    y_test_info = (
        pd.DataFrame(np.unique(decoded_y_test, return_counts=True)).T.set_index(0)
        / len(decoded_y_test)
        * 100
    )
    y_test_info_raw = pd.DataFrame(
        np.unique(decoded_y_test, return_counts=True)
    ).T.set_index(0)
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
    y_train_test_stats.index = y_encoder.inverse_transform(
        np.eye(len(y_encoder.classes_))
    )
    return y_train_test_stats


def knn_train_predict(
    X_train, X_test, y_train, y_test, y_encoder, n_neighbors=1
) -> np.array:
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def mlp_train_predict(
    X_train, X_test, y_train, y_test, y_encoder, **kwargs
) -> np.array:
    """Create a simple MLP. Train data on it, then predict on it.

    Args:
        X_train (np.array): X training data.
        X_test (np.array): X testing data.
        y_train (np.array): y training data.
        y_test (np.array): y testing data. Not used.
        y_encoder (any): encoder for y. Not used.

    Returns:
        np.array: predicted results.
    """
    model = MLPClassifier(
        random_state=1, max_iter=300, early_stopping=False, **kwargs
    ).fit(X_train, y_train)
    return model.predict(X_test)


def tree_train_predict(
    X_train, X_test, y_train, y_test, y_encoder, **kwargs
) -> np.array:
    clf = DecisionTreeClassifier(random_state=555, **kwargs).fit(X_train, y_train)
    return clf.predict(X_test)
