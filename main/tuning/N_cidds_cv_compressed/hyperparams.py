import keras_tuner as kt


DEFAULT_HYPERPARAMS_TO_TUNE = {
    "discriminator": {
        "hidden_layer_width": 128,
        "learning_rate": 1e-5,
        "beta_1": 0.5,
    },
    "generator": {
        "hidden_layer_width": 128,
        "learning_rate": 1e-5,
        "beta_1": 0.5,
    },
    "hidden_layer_depth": 5,
    "num_epochs": 10,
    "batch_size": 128,
    "latent_dim": 100,
}


def recursive_dict_union(dict1, dict2):
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = recursive_dict_union(result[key], value)
        else:
            result[key] = value

    return result


def get_hyperparams_to_tune(hp: kt.HyperParameters):
    hyperparams_to_tune = {
        "discriminator": {
            "hidden_layer_width": hp.Int(
                "d_hidden_layer_width", min_value=256, max_value=256 * 4, step=256
            ),
            "learning_rate": hp.Choice(
                "d_learning_rate", values=[3e-4, 8e-4, 3e-5, 8e-5, 3e-6]
            ),
            "beta_1": hp.Choice("d_beta_1", values=[0.0, 0.5, 0.9]),
        },
        "generator": {
            "hidden_layer_width": hp.Int(
                "g_hidden_layer_width", min_value=256, max_value=256 * 4, step=256
            ),
            "learning_rate": hp.Choice(
                "g_learning_rate", values=[1e-4, 5e-4, 1e-5, 5e-5, 1e-6]
            ),
            "beta_1": hp.Choice("g_beta_1", values=[0.0, 0.5, 0.9]),
        },
        "num_epochs": hp.Choice("num_epochs", values=[5, 10, 15, 20, 50, 70, 100]),  # ! change when done debugging
        "batch_size": hp.Choice(
            "batch_size", values=[128, 256, 1024, 4096, 8192, 16384, 32768]
        ),
        "latent_dim": hp.Choice(
            "latent_dim", values=[256, 512]
        ),
        "hidden_layer_depth": hp.Int(
            "hidden_layer_depth", min_value=3, max_value=6, step=1
        ),
    }
    all_hps = recursive_dict_union(DEFAULT_HYPERPARAMS_TO_TUNE, hyperparams_to_tune)
    return all_hps
