import numpy as np
import tensorflow as tf
from typing import List, Dict
from sklearn.model_selection import KFold


def get_train_test_datasets() -> List[Dict]:
    # Set a seed for reproducibility
    np.random.seed(42)

    # Generate 5,000 samples for each label
    num_samples_per_label = 5000

    # Generate data with label 0 (y = 0.8 + small noise)
    X_label_0 = np.random.rand(num_samples_per_label, 1)  # Feature 1
    X_label_0 = np.concatenate(
        [X_label_0, 0.8 + 0.03 * np.random.randn(num_samples_per_label, 1)], axis=1
    )  # Feature 2
    y_label_0 = np.zeros((num_samples_per_label, 1))

    # Generate data with label 1 (y = 0.2 + small noise)
    X_label_1 = np.random.rand(num_samples_per_label, 1)  # Feature 1
    X_label_1 = np.concatenate(
        [X_label_1, 0.2 + 0.03 * np.random.randn(num_samples_per_label, 1)], axis=1
    )  # Feature 2
    y_label_1 = np.ones((num_samples_per_label, 1))

    just_data = np.vstack((X_label_0, X_label_1)).astype(np.float32)
    just_labels = np.vstack((y_label_0, y_label_1)).astype(np.float32)
    entire_dataset = np.hstack((just_data, just_labels))

    # split into 3 folds
    folds = []
    kf = KFold(n_splits=3)
    kf.get_n_splits(entire_dataset)
    for train_index, test_index in kf.split(entire_dataset):
        X_y_train, X_y_test = entire_dataset[train_index], entire_dataset[test_index]
        # Create a TensorFlow Dataset for training data
        train_dataset = tf.data.Dataset.from_tensor_slices(X_y_train)
        # split testing data into X and y
        X_test = X_y_test[:, :2]
        y_test = X_y_test[:, 2]
        # assemble as dictionary
        folds.append(
            {"train_dataset": train_dataset, "X_test": X_test, "y_test": y_test}
        )
    return folds
