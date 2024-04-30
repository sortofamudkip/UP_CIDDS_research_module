import pandas as pd
from pathlib import Path
import json
import numpy as np

def read_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def process_trial_json(trial_json:dict):
    hps = trial_json["hyperparameters"]["values"]
    result = trial_json["metrics"]["metrics"]["default_objective"]["observations"][0]["value"][0]
    hps["method"] = "N"
    hps["result"] = result
    return hps

def get_results():

    this_dir = Path(__file__).parent
    # find all directories that start with "hp_tuning_dir"
    hp_tuning_dirs = [x for x in this_dir.iterdir() if x.is_dir() and x.name.startswith("hp_tuning_dir")]

    trial_dfs : list = []
    for hp_tuning_dir in hp_tuning_dirs:
        # get the first folder in the directory (there's only one)
        hp_tuning_dir = next(hp_tuning_dir.iterdir())
        # find all folders that start with "trial_"
        trial_jsons = [read_json(x / "trial.json") for x in hp_tuning_dir.iterdir() if x.is_dir() and x.name.startswith("trial_")]
        # process all the jsons
        trial_jsons = [process_trial_json(x) for x in trial_jsons]
        # convert to dataframe
        trial_df = pd.DataFrame(trial_jsons)
        trial_dfs.append(trial_df)

    # convert to dataframe
    df = pd.concat(trial_dfs)
    # rename cols
    df = df.rename(columns={"d_hidden_layer_width": "d width", "d_learning_rate": "d lr", "d_hidden_layer_depth": "d depth", "d_beta_1": "d beta", "g_hidden_layer_width": "g width", "g_learning_rate": "g lr", "g_hidden_layer_depth": "g depth", "g_beta_1": "g beta", "num_epochs": "epochs", "batch_size": "batch_size"})
    df["result"] = -df["result"]
    df.loc[(df["result"] == -np.inf) | (df["result"] == np.inf), "result"] = np.nan
    df["method"] = "N"
    # sort by result
    df = df.sort_values(by="result", ascending=False)
    return df