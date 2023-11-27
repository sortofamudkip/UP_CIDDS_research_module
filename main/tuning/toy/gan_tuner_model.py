import tensorflow as tf
import keras_tuner as kt
import numpy as np
from toy_gan import WCGAN_GP
from hyperparams import get_hyperparams_to_tune, DEFAULT_HYPERPARAMS_TO_TUNE


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
        # ^ don't touch this until this code can run!!
        return np.random.uniform(low=0, high=1, size=1)[0]

    def fit(
        self,
        hp: kt.HyperParameters(),  # mandatory?
        model: WCGAN_GP,  # mandatory?
        real_dataset: tf.data.Dataset,  # big tensor with ALL the data [X,y]
        X_test: np.array,
        y_test: np.array,
        **kwargs: dict,  # for epochs, batch_size, etc.
    ):
        # train the gan model
        model.fit(real_dataset, **kwargs)
        # evaluate the gan model
        tstr_score = self.evaluate_TSTR(model, X_test, y_test)
        return tstr_score


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
