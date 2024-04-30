from preprocess_data import decode_N_WGAN_GP
import tensorflow as tf
from load_data import get_datasets_and_info
from gan_tuner_model import GANTunerModelCV


# from hyperparams import get_hyperparams_to_tune
import keras_tuner as kt
import logging
import random

logging.basicConfig(level=logging.DEBUG)

# create a random token of length 5
random_token = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))
project_dir = f"hp_tuning_dir_🐟{random_token}"
project_name = f"cidds_wcgan_gp_🐟{random_token}"
logging.info(f"random_token: {random_token}")


# get train and test datasets
dataset_info = get_datasets_and_info()

# output dim is the number of features (WITHOUT the labels)
output_dim = dataset_info["folds"][0]["X_test"].shape[1]
X_encoders = dataset_info["X_encoders"]
y_encoder = dataset_info["y_encoder"]
X_colnames = dataset_info["X_colnames"]

# define the gan model (hp is defined inside this class)
hypermodel = GANTunerModelCV(
    output_dim=output_dim,
    num_classes=2,
    X_encoders=X_encoders,
    y_encoder=y_encoder,
    X_colnames=X_colnames,
    decoder_func=decode_N_WGAN_GP,
)


# instantiate the tuner
tuner = kt.RandomSearch(
    hypermodel=hypermodel,
    # no objective because it's the return value of `HyperModel.fit()`;
    # if it doesn't work, try https://keras.io/guides/keras_tuner/getting_started/#custom-metric-as-the-objective
    max_trials=5,  # ! THIS NUMBER IS THE MOST IMPORTANT. DON'T MAKE IT TOO HIGH!
    executions_per_trial=1,  # * how many times to train a model with the same hps
    overwrite=True,  # * overwrite previous results
    directory=project_dir,
    project_name=project_name,
    max_consecutive_failed_trials=100, # basically infinite
    # distribution_strategy=tf.distribute.MirroredStrategy(),
)

tuner.search(
    dataset_info=dataset_info,
)

tuner.results_summary()
best_hps = tuner.get_best_hyperparameters()[0]
print("Best values:", best_hps.values)

# best_model = tuner.get_best_models()[0]
# best_model.summary()
