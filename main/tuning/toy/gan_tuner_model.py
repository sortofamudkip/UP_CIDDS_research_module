import logging
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from toy_gan import WCGAN_GP, StopTrainingOnNaNCallback
from hyperparams import get_hyperparams_to_tune, DEFAULT_HYPERPARAMS_TO_TUNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


class GANTunerModel(kt.HyperModel):
    def __init__(self, output_dim: int, num_classes: int):
        self.output_dim = output_dim
        self.num_classes = num_classes

    def build(self, hp: kt.HyperParameters):
        # define the gan model with the hyperparameters to tune
        self.hyperparams_to_tune = get_hyperparams_to_tune(hp)
        model_gan = WCGAN_GP(
            self.output_dim, self.num_classes, self.hyperparams_to_tune
        )
        # compile the gan model
        model_gan.compile()
        return model_gan

    def evaluate_TSTR(self, model: WCGAN_GP, X_test: np.array, y_test: np.array):
        # TSTR: train synthetic, test real
        # The GAN produces synthetic samples which are then used to train a classifier,
        #   then the classifier is evaluated on real data.
        # ^ Create a dataset with the synthetic samples
        num_samples = X_test.shape[0]
        ## + Generate random labels from unifrom distribution
        y_fake: tf.Tensor = model.generate_fake_labels(num_samples)
        ## + Generate synthetic samples
        X_fake = model.generate_fake_samples_without_labels(num_samples, y_fake)
        ## + Convert these two into numpy arrays
        X_fake = X_fake.numpy()
        y_fake = y_fake.numpy()
        ## & if X_fake contains NaNs, warn the user and return positive infinity
        if np.isnan(X_fake).any():
            logging.warning("X_fake contains NaNs")
            return np.inf
        # ^ Train a classifier on the synthetic samples
        clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=11)
        clf.fit(X_fake, y_fake.ravel())

        # ^ Evaluate classifier on real data (X_test, y_test) to get TSTR score
        # predict on real test data
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logging.info(f"-----Results-----")
        logging.info(f"Hyperparams: {self.hyperparams_to_tune}")
        logging.info(f"Accuracy: {acc}, F1: {f1}")
        logging.info(f"-----------------")
        return acc

    def fit(
        self,
        hp: kt.HyperParameters(),  # mandatory
        model: WCGAN_GP,  # mandatory
        real_dataset: tf.data.Dataset,  # big tensor with ALL the data [X,y]
        X_test: np.array,
        y_test: np.array,
        callbacks: list = None,
        **kwargs: dict,  # for epochs, batch_size, etc.
    ):
        # train the gan model
        hp_num_epochs = self.hyperparams_to_tune["num_epochs"]
        hp_batch_size = self.hyperparams_to_tune["batch_size"]
        model.fit(
            real_dataset.batch(hp_batch_size),
            epochs=hp_num_epochs,
            verbose=0,
            callbacks=[StopTrainingOnNaNCallback()],
            **kwargs,
        )
        # evaluate the gan model
        tstr_score = self.evaluate_TSTR(model, X_test, y_test)
        return -tstr_score  # we want to maximize this score, so we return the negative


"""
Main idea:
1. ✅ Define 'build' in HyperGAN class 
2. Define 'fit' in HyperGAN class
2.5. Define evaluation metric in HyperGAN class
3. Instantiate HyperGAN class as "model"
4. Instantiate tuner, e.g. kt.RandomSearch (use tuner.search_space_summary() to check the search space)
5. Create dataset
6. Run tuner.search() to find the best hyperparameters
7. Retrieve the best hyperparameters using tuner.get_best_hyperparameters()
"""
