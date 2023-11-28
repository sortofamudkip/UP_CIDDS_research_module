import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle

preprocessed_data_dir = (
    Path(__file__).parent.parent.parent.parent.parent
    / "preprocessed/external/N/crossval"
)

fold_fnames = [
    (
        preprocessed_data_dir / "X_y_2c_N_fold1_train",
        preprocessed_data_dir / "X_y_2c_N_fold1_test",
    ),
    (
        preprocessed_data_dir / "X_y_2c_N_fold2_train",
        preprocessed_data_dir / "X_y_2c_N_fold2_test",
    ),
    (
        preprocessed_data_dir / "X_y_2c_N_fold3_train",
        preprocessed_data_dir / "X_y_2c_N_fold3_test",
    ),
]
X_encoder_fname = preprocessed_data_dir / "X_encoders"


def _get_balanced_dataset(dataset_np: np.array) -> tf.data.Dataset:
    """
    Returns a balanced dataset as a TensorFlow dataset object. Binary classification only.

    Args:
    - dataset_np: A numpy array containing the dataset (both X and y).

    Returns:
    - A TensorFlow dataset object containing the balanced dataset.
    """
    raveled_y = dataset_np[:, -1].ravel()  # last column is y
    # split into real and fake
    normal_samples = dataset_np[raveled_y == 1, :]  # 1 is normal
    attacker_samples = dataset_np[raveled_y == 0, :]  # 0 is attacker
    # turn into td.data.Dataset
    normal_dataset = tf.data.Dataset.from_tensor_slices(normal_samples)
    attacker_dataset = tf.data.Dataset.from_tensor_slices(attacker_samples)
    # use tf.data.Dataset.sample_from_datasets to sample from both datasets
    dataset = tf.data.Dataset.sample_from_datasets(
        [normal_dataset, attacker_dataset], [0.5, 0.5]
    )
    return dataset


def get_datasets_and_info() -> dict:
    # split into 3 folds
    folds = []
    for train_pickle_fname, test_pickle_fname in fold_fnames:
        with open(train_pickle_fname, "rb") as f:
            X_train, y_train, y_encoder, X_colnames, X_train_encoders = pickle.load(f)
        # * train dataset
        X_y_train = np.hstack([X_train, y_train]).astype(np.float32)
        # Create a TensorFlow Dataset for training data
        train_dataset = _get_balanced_dataset(X_y_train)

        # * test dataset
        with open(test_pickle_fname, "rb") as f:
            X_test, y_test, y_encoder, X_colnames, X_test_encoders = pickle.load(f)
        # y_test = y_encoder.inverse_transform(y_test.ravel())
        # * read X_encoders
        with open(X_encoder_fname, "rb") as f:
            X_encoders = pickle.load(f)

        # assemble as dictionary
        folds.append(
            {
                "train_dataset": train_dataset,
                "X_test": X_test,
                "y_test": y_test,
            }
        )
    all_info = {
        "X_colnames": X_colnames,
        "X_encoders": X_encoders,
        "folds": folds,
        "y_encoder": y_encoder,
    }
    return all_info
