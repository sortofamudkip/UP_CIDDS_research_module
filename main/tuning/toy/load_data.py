import numpy as np
import tensorflow as tf
from typing import Tuple


def get_train_dataset() -> tf.data.Dataset:
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

    # Create a TensorFlow Dataset
    dataset = (
        tf.data.Dataset.from_tensor_slices(entire_dataset).shuffle(10000).batch(64)
    )
    return dataset


def get_test_dataset() -> Tuple[np.ndarray, np.ndarray]:
    # Set a seed for reproducibility
    np.random.seed(1234)

    # Generate samples for each label
    num_samples_per_label = 1000

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

    return just_data, just_labels
