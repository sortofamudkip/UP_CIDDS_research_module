import keras_tuner as kt


DEFAULT_HYPERPARAMS_TO_TUNE = {
    "discriminator": {
        "hidden_layer_width": 128,
        "learning_rate": 0.0002,
    },
    "generator": {
        "hidden_layer_width": 128,
        "learning_rate": 0.0002,
    },
    "num_epochs": 10,
    "batch_size": 128,
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
                "hidden_layer_width", min_value=128, max_value=128 * 3, step=128
            ),
            "learning_rate": hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5]),
        },
        "generator": {
            "hidden_layer_width": hp.Int(
                "hidden_layer_width", min_value=128, max_value=128 * 3, step=128
            ),
            "learning_rate": hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5]),
        },
        "num_epochs": hp.Int("num_epochs", min_value=10, max_value=100, step=30),
        "batch_size": hp.Int("batch_size", min_value=128, max_value=256, step=64),
    }
    all_hps = recursive_dict_union(DEFAULT_HYPERPARAMS_TO_TUNE, hyperparams_to_tune)
    return all_hps
