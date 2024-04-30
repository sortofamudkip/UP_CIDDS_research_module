import numpy as np
import tensorflow as tf
from pathlib import Path
import pickle

preprocessed_data_dir = (
    Path(__file__).parent.parent.parent.parent
    / "preprocessed_compressed/external/N/crossval"
).resolve()

fold_fnames = [preprocessed_data_dir / f"X_y_2c_N_fold{i+1}.npz" for i in range(3)]
X_encoder_fname = preprocessed_data_dir / "X_encoders"
X_colnames_fname = preprocessed_data_dir / "X_colnames"
y_encoder_fname = preprocessed_data_dir / "y_encoder"


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
    # upsample attacker samples
    num_normal_samples = normal_samples.shape[0]
    num_attacker_samples = attacker_samples.shape[0]
    num_normal_to_attacker_ratio = int(num_normal_samples / num_attacker_samples)
    
    # turn into td.data.Dataset
    normal_dataset = tf.data.Dataset.from_tensor_slices(normal_samples)
    attacker_dataset = tf.data.Dataset.from_tensor_slices(attacker_samples.repeat(num_normal_to_attacker_ratio, axis=0))
    # use tf.data.Dataset.sample_from_datasets to sample from both datasets
    dataset = tf.data.Dataset.sample_from_datasets(
        [normal_dataset, attacker_dataset], [0.5, 0.5]
    )
    return dataset


def get_datasets_and_info() -> dict:
    # split into 3 folds
    folds = []
    for npz_name in fold_fnames:
        # * read numpy array
        arrays = np.load(npz_name)
        X_train = arrays["X_train"]
        y_train = arrays["y_train"]
        X_test = arrays["X_test"]
        y_test = arrays["y_test"]
        # * create train dataset
        X_y_train = np.hstack([X_train, y_train]).astype(np.float32)
        # Create a TensorFlow Dataset for training data
        train_dataset = _get_balanced_dataset(X_y_train)

        # * read X_encoders
        with open(X_encoder_fname, "rb") as f:
            X_encoders = pickle.load(f)
        # * read X_colnames
        with open(X_colnames_fname, "rb") as f:
            X_colnames = pickle.load(f)
        # * read y_encoder
        with open(y_encoder_fname, "rb") as f:
            y_encoder = pickle.load(f)

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

if __name__ == "__main__":
    data = get_datasets_and_info()
    print(data)