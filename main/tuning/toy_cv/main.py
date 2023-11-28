from site import execusercustomize
from toy_gan import WCGAN_GP
from load_data import get_train_test_datasets
from gan_tuner_model import GANTunerModelCV
from hyperparams import get_hyperparams_to_tune
import keras_tuner as kt
import logging

logging.basicConfig(level=logging.DEBUG)


# get train and test datasets
dataset_folds = get_train_test_datasets()

# output dim is the number of features (WITHOUT the labels)
output_dim = dataset_folds[0]["X_test"].shape[1]

# define the gan model (hp is defined inside this class)
hypermodel = GANTunerModelCV(
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
    overwrite=True,  # * overwrite previous results
    directory="hp_tuning_toy_dir",
    project_name="toy_wcgan_gp",
)

tuner.search(
    dataset_folds=dataset_folds,
)

tuner.results_summary()
best_hps = tuner.get_best_hyperparameters()[0]
print("Best values:", best_hps.values)

# best_model = tuner.get_best_models()[0]
# best_model.summary()
