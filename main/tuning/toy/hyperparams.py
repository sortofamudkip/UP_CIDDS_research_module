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
                "hidden_layer_width", min_value=64, max_value=512, step=64
            ),
            # "learning_rate": hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4]),
        },
        "generator": {
            # "hidden_layer_width": hp.Int(
            #     "hidden_layer_width", min_value=64, max_value=512, step=64
            # ),
            # "learning_rate": hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4]),
        },
    }
    return hyperparams_to_tune
