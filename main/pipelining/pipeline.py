from contextlib import redirect_stdout
import json
from pathlib import Path
import numpy as np

import pandas as pd
from .pipeline_2classes import (
    eval_2classes,
    load_testdata_2classes,
)
from .pipeline_5classes import (
    eval_5classes,
    load_testdata_5classes,
)
from ..synthetic_eval.evaluate_synthetic import eval_synthetic_one_epoch
from sys import stdout


from pathlib import Path


def create_output_dir(pipeline_name: str, skip_if_exists=False) -> Path:
    """
    Creates a new directory for the pipeline's output files.

    Args:
        pipeline_name (str): The name of the pipeline.
        skip_if_exists (bool): If True, returns the existing directory instead of creating a new one.

    Returns:
        Path: The path to the newly created directory.
    """
    output_dir = (Path(__file__).parent / "../../../results" / pipeline_name).resolve()
    if skip_if_exists and output_dir.is_dir():
        return output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        print(f"Created output dir: {output_dir}")
    except FileExistsError:
        print(f"Dir already exists: {output_dir} ")
        assert False
    return output_dir


def load_testdata(test_data_pickle_fname: str, num_classes: int):
    """
    Load test data from a pickle file and return the input and target data.

    Args:
        test_data_pickle_fname (str): The filename of the pickle file containing the test data.
        num_classes (int): The number of classes in the dataset. Must be either 2 or 5.

    Returns:
        tuple: A tuple containing the input and target data as numpy arrays.
    """
    assert num_classes in (2, 5)
    X_test, y_test = (
        load_testdata_2classes(test_data_pickle_fname)
        if num_classes == 2
        else load_testdata_5classes(test_data_pickle_fname)
    )
    return X_test, y_test


def eval_dataset(
    pipeline_name: str,
    num_epochs: int,
    gan_pipeline,
    X_test: np.array,
    y_test: np.array,
    num_classes: int,
) -> pd.DataFrame:
    """
    Evaluate a GAN pipeline on a given test dataset.

    Args:
        pipeline_name (str): Name of the GAN pipeline being evaluated.
        num_epochs (int): Number of epochs the GAN pipeline was trained for.
        gan_pipeline: The GAN pipeline being evaluated.
        X_test (np.array): Test dataset features.
        y_test (np.array): Test dataset labels.
        num_classes (int): Number of classes in the dataset. Must be 2 or 5.

    Returns:
        pd.DataFrame: A summary of the evaluation results.
    """

    assert num_classes in (2, 5), "Number of classes can only be 2 or 5"
    summary_df = (
        eval_2classes(pipeline_name, num_epochs, gan_pipeline, X_test, y_test)
        if num_classes == 2
        else eval_5classes(pipeline_name, num_epochs, gan_pipeline, X_test, y_test)
    )
    return summary_df


def generate_and_eval_dataset_once(
    gan_pipeline,
    X_test: np.array,
    y_test: np.array,
    num_classes: int,
) -> pd.DataFrame:
    assert num_classes in (2, 5), "Number of classes can only be 2 or 5"
    num_rows_to_generate = gan_pipeline.X.shape[0]
    generated_samples = gan_pipeline.generate_samples(num_rows_to_generate)
    summary_df = eval_synthetic_one_epoch(
        gan_pipeline, generated_samples, X_test, y_test, num_classes
    )
    return summary_df


import json
import pandas as pd
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Any, Tuple

from .preprocessor import load_testdata, decode_N_WGAN_GP
from .basic_gan import BasicGANPipeline


def run_pipeline(
    pipeline_name: str,  # ex: "N_2_25epochs_TSTR001"
    train_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_train"
    test_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_test"
    num_epochs: int,  # ex: 5
    gan_pipeline_class,  # ex: basic_gan.BasicGANPipeline
    preprocessor_function,  # ex: preprocessor.decode_N_WGAN_GP
    num_classes: int,  # ex: 2
    batch_size: int = 1024,
    learning_rate: float = 0.00001,
    fold: str = "",  # used to give different file names for crossval
    latent_dim: int = 0,  # ^ since latent size is also a hyperparam
) -> Dict[str, Any]:
    """
    Runs a GAN pipeline with the given parameters.

    Args:
        pipeline_name (str): Name of the pipeline.
        train_data_pickle_fname (str): Filepath of the preprocessed training data.
        test_data_pickle_fname (str): Filepath of the preprocessed test data.
        num_epochs (int): Number of epochs to train the GAN for.
        gan_pipeline_class (class): Class of the GAN pipeline to use.
        preprocessor_function (function): Function to use for preprocessing the data.
        num_classes (int): Number of classes in the data.
        batch_size (int, optional): Batch size to use for training. Defaults to 1024.
        learning_rate (float, optional): Learning rate to use for training. Defaults to 0.00001.
        fold (str, optional): Used to give different file names for cross-validation. Defaults to "".
        latent_dim (int, optional): Size of the latent dimension. Defaults to 0.

    Returns:
        dict: A dictionary containing the summary dataframe, the trained GAN, and the losses.
    """

    # basic checks
    assert num_classes == 2 or num_classes == 5

    # make files with folds look nicer
    if fold and not fold.startswith("_"):
        fold = f"fold_{fold}"

    output_dir = create_output_dir(pipeline_name, bool(fold))
    with open(output_dir / f"log{fold}.txt", "w") as f:
        with redirect_stdout(f):
            # with redirect_stdout(stdout): # ^ for debugging
            # Load data & init pipline
            print("Loading training data and initialising GAN...")
            gan_pipeline = gan_pipeline_class(
                train_data_pickle_fname,
                preprocessor_function,
                pipeline_name,
                subset=False,
                batch_size=batch_size,
                latent_dim=latent_dim,
            )

            # Save summary of parameters
            with open(f"results/{pipeline_name}/params.json", "w") as f:
                params = {
                    "pipeline_name": pipeline_name,
                    "train_data_pickle_fname": train_data_pickle_fname,
                    "test_data_pickle_fname": test_data_pickle_fname,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "preprocessing_method": str(preprocessor_function),
                    "num_classes": num_classes,
                }
                json.dump(params, f, indent=4)
            print(f"Parameters summary saved in results/{pipeline_name}/params.json")

            # train GAN
            print("Training GAN...")
            gan_history = gan_pipeline.compile_and_fit_GAN(
                learning_rate=learning_rate, beta_1=0.90, epochs=num_epochs
            )

            # plot and save losses
            print("Plotting and saving losses...")
            losses = pd.DataFrame(
                {
                    "g_loss": gan_history.history["g_loss"],
                    "d_loss": gan_history.history["d_loss"],
                }
            ).rename_axis("Epoch")
            losses.to_csv(f"results/{pipeline_name}/losses{fold}.csv")
            losses.plot(title="GAN Losses", ylabel="loss").get_figure().savefig(
                f"results/{pipeline_name}/losses{fold}.jpg"
            )

            # load test data
            print("Loading test data...")
            X_test, y_test = load_testdata(test_data_pickle_fname, num_classes)

            # generate and evaluate synthetic data
            print("Evaluating synthetic data...")
            summary_df = generate_and_eval_dataset_once(
                gan_pipeline, X_test, y_test, num_classes
            )

            # save summary data
            with open(f"results/{pipeline_name}/metrics{fold}.json", "w") as f:
                json.dump(summary_df, f, indent=4)
    return {"summary_df": summary_df, "gan": gan_pipeline, "losses": losses}
