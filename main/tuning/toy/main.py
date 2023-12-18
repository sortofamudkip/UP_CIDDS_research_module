from site import execusercustomize
from toy_gan import WCGAN_GP
from load_data import get_train_dataset, get_test_dataset
from gan_tuner_model import GANTunerModel
from hyperparams import get_hyperparams_to_tune
import keras_tuner as kt
import logging
import random

logging.basicConfig(level=logging.DEBUG)

# create a random token of length 5
random_token = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))
project_dir = f"hp_tuning_dir_{random_token}"
project_name = f"toy_wcgan_gp_{random_token}"
logging.info(f"random_token: {random_token}")

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
    max_trials=2,  # ! max amount of hyperparameter combinations to try; INCREASE THIS LATER
    executions_per_trial=1,  # * how many times to train a model with the same hps
    overwrite=True,  # * overwrite previous results
    directory=project_dir,
    project_name=project_name,
)

tuner.search(
    real_dataset=train_dataset,
    X_test=X_test,
    y_test=y_test,
)

tuner.results_summary()
best_hps = tuner.get_best_hyperparameters()[0]
print("Best values:", best_hps.values)

# best_model = tuner.get_best_models()[0]
# best_model.summary()
