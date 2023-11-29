import logging
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from gan import CIDDS_WCGAN_GP, StopTrainingOnNaNCallback
from hyperparams import get_hyperparams_to_tune, DEFAULT_HYPERPARAMS_TO_TUNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Dict
from preprocess_data import decode_N_WGAN_GP


class GANTunerModelCV(kt.HyperModel):
    def __init__(
        self,
        output_dim: int,
        num_classes: int,
        X_encoders: dict,
        y_encoder,
        X_colnames: list,
        decoder_func: callable,  # e.g. decode_N_WGAN_GP
    ):
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.X_encoders = X_encoders
        self.y_encoder = y_encoder
        self.X_colnames = X_colnames
        self.decoder_func = decoder_func

    def build(self, hp: kt.HyperParameters):
        # define the gan model with the hyperparameters to tune
        self.hyperparams_to_tune = get_hyperparams_to_tune(hp)
        model_gan = CIDDS_WCGAN_GP(
            output_dim=self.output_dim,
            num_classes=self.num_classes,
            x_col_labels=self.X_colnames,
            x_encoders=self.X_encoders,
            decoder_func=self.decoder_func,
            y_encoder=self.y_encoder,
            hyperparams_to_tune=self.hyperparams_to_tune,
        )
        # compile the gan model
        model_gan.compile()
        return model_gan

    def evaluate_TSTR(self, model: CIDDS_WCGAN_GP, X_test: np.array, y_test: np.array):
        # TSTR: train synthetic, test real
        # The GAN produces synthetic samples which are then used to train a classifier,
        #   then the classifier is evaluated on real data.
        # ^ Create a dataset with the synthetic samples
        num_samples = X_test.shape[0]
        attacker_label = self.y_encoder.transform(["attacker"])[0][0]
        ## + generate n plausible samples
        X_y_fake, retention_scores = model.generate_n_plausible_samples(
            n_x_rows=num_samples, n_target_rows=num_samples
        )
        X_fake = X_y_fake[:, :-1]
        y_fake = X_y_fake[:, -1]
        ## & if X_fake contains NaNs, warn the user and return positive infinity
        if np.isnan(X_fake).any():
            logging.warning(
                "X_fake contains NaNs or generated too many implausible samples"
            )
            return np.inf
        # ^ Train a classifier on the synthetic samples
        clf = RandomForestClassifier(random_state=0, n_estimators=11)
        clf.fit(X_fake, y_fake.ravel())

        # ^ Evaluate classifier on real data (X_test, y_test) to get TSTR score
        # predict on real test data
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=attacker_label)
        logging.info(f"Accuracy: {acc}, F1: {f1}")
        return acc

    def fit(
        self,
        hp: kt.HyperParameters(),  # mandatory
        model: CIDDS_WCGAN_GP,  # mandatory
        dataset_info: dict,
        callbacks: list = None,
        **kwargs: dict,  # for epochs, batch_size, etc.
    ):
        # train the gan model
        hp_num_epochs = self.hyperparams_to_tune["num_epochs"]
        hp_batch_size = self.hyperparams_to_tune["batch_size"]

        # for each fold in dataset_folds, obtain TSTR score and return the average
        tstr_scores = []
        X_colnames = dataset_info["X_colnames"]
        X_encoders = dataset_info["X_encoders"]
        dataset_folds = dataset_info["folds"]
        for i, fold in enumerate(dataset_folds):
            # unpack the fold
            real_dataset = fold["train_dataset"]
            X_test = fold["X_test"]
            y_test = fold["y_test"]
            # train the model
            model.fit(
                real_dataset.batch(hp_batch_size),
                epochs=hp_num_epochs,
                verbose=0,
                callbacks=[StopTrainingOnNaNCallback()],
                **kwargs,
            )
            # evaluate the gan model
            tstr_score = self.evaluate_TSTR(model, X_test, y_test)
            tstr_scores.append(tstr_score)

        # return the average TSTR score
        avg_tstr_score = np.mean(tstr_scores)
        return -avg_tstr_score  # * negative because kt maximizes the objective


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
