from site import execusercustomize
from toy_gan import WCGAN_GP
from load_data import get_train_dataset, get_test_dataset
from gan_tuner_model import GANTunerModel
from hyperparams import get_hyperparams_to_tune
import keras_tuner as kt
import logging

logging.basicConfig(level=logging.DEBUG)


# get train and test datasets
train_dataset = get_train_dataset()
X_test, y_test = get_test_dataset()

# output dim is the number of features (WITHOUT the labels)
output_dim = X_test.shape[1]

# define the gan model (hp is defined inside this class)
hypermodel = GANTunerModel(
    output_dim=output_dim,
    num_classes=2,
)


# instantiate the tuner
tuner = kt.RandomSearch(
    hypermodel=hypermodel,
    # no objective because it's the return value of `HyperModel.fit()`;
    # if it doesn't work, try https://keras.io/guides/keras_tuner/getting_started/#custom-metric-as-the-objective
    max_trials=5,  # ! max amount of hyperparameter combinations to try; INCREASE THIS LATER
    executions_per_trial=1,  # * how many times to train a model with the same hps
    overwrite=True,
    directory="hp_tuning_toy_dir",
    project_name="toy_wcgan_gp",
)

tuner.search(
    real_dataset=train_dataset,
    X_test=X_test,
    y_test=y_test,
    epochs=3,
)

print(tuner.results_summary())
