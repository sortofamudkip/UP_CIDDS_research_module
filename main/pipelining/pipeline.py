from contextlib import redirect_stdout
from doctest import debug
import json
from pathlib import Path
import numpy as np
import logging

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
from typing import Dict, Any, Tuple

from ..gan_models.basic_gan import BasicGANPipeline


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
        logging.info(f"Created output dir: {output_dir}")
    except FileExistsError:
        logging.error(f"Dir already exists: {output_dir} ")
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
    gan_pipeline: BasicGANPipeline,
    X_test: np.array,
    y_test: np.array,
    num_classes: int,
    synthetic_to_real_ratio: float = -1,
) -> pd.DataFrame:
    """
    Generates synthetic samples using the provided GAN pipeline and evaluates them against the provided test set.

    Args:
        gan_pipeline (GANPipeline): The GAN pipeline to use for generating synthetic samples.
        X_test (np.array): The feature matrix of the test set.
        y_test (np.array): The target vector of the test set.
        num_classes (int): The number of classes in the target vector. Must be either 2 or 5.
        synthetic_to_real_ratio (int): The ratio of synthetic to real data to use for evaluation.
                                       If -1, use all synthetic data. Defaults to -1.

    Returns:
        pd.DataFrame: A summary of the evaluation results.
    """
    assert num_classes in (2, 5), "Number of classes can only be 2 or 5"

    # if T(S+R)TR: add real data to synthetic data
    if synthetic_to_real_ratio > 0:
        logging.info(f"Using T(S+R)TR with ratio |S| ={synthetic_to_real_ratio}|R|")
        num_rows_to_generate = int(gan_pipeline.X.shape[0] * synthetic_to_real_ratio)
        generated_samples, _ = gan_pipeline.generate_n_plausible_samples(
            num_rows_to_generate
        )
        real_data = np.hstack((gan_pipeline.X, gan_pipeline.y))  # real X and y
        generated_samples = np.vstack((generated_samples, real_data))
        np.random.shuffle(generated_samples)  # shuffle

    # else: TSTR, only use S
    else:
        # generate synthetic data
        num_rows_to_generate = gan_pipeline.X.shape[0]
        logging.info(
            f"Using TSTR, |S| = {num_rows_to_generate}/{gan_pipeline.X.shape[0]}|R|"
        )
        generated_samples, _ = gan_pipeline.generate_n_plausible_samples(
            num_rows_to_generate
        )

    # evaluate synthetic data
    summary_df = eval_synthetic_one_epoch(
        gan_pipeline, generated_samples, X_test, y_test, num_classes
    )
    return summary_df


def run_pipeline(
    pipeline_name: str,  # ex: "N_2_25epochs_TSTR001"
    train_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_train"
    test_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_test"
    num_epochs: int,  # ex: 5
    gan_pipeline_class: BasicGANPipeline,  # ex: basic_gan.BasicGANPipeline
    preprocessor_function,  # ex: preprocessor.decode_N_WGAN_GP
    num_classes: int,  # ex: 2
    batch_size: int = 1024,
    d_learning_rate: float = 0.0003,
    g_learning_rate: float = 0.0001,
    fold: str = "",  # used to give different file names for crossval
    latent_dim: int = 0,  # ^ since latent size is also a hyperparam
    synthetic_to_real_ratio: float = -1,  # ^ for T(S+R)TR
    use_balanced_dataset: bool = True,  # whether to use a balanced dataset for training or not
    is_debug: bool = False,
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
        synthetic_to_real_ratio (float, optional): Ratio of synthetic to real data to use for evaluation. Defaults to -1.

    Returns:
        dict: A dictionary containing the summary dataframe, the trained GAN, and the losses.
    """

    # basic checks
    assert num_classes == 2 or num_classes == 5

    # make files with folds look nicer
    if fold and not fold.startswith("_"):
        fold = f"fold_{fold}"

    output_dir = create_output_dir(pipeline_name, bool(fold))
    output_file_name = str((output_dir / f"log{fold}.log").resolve())
    logging.basicConfig(
        filename=output_file_name if not is_debug else None,
        encoding="utf-8",
        level=logging.INFO if not is_debug else logging.DEBUG,
        force=True,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load data & init pipline
    logging.info("Loading training data and initialising GAN...")
    gan_pipeline = gan_pipeline_class(
        train_data_pickle_fname,
        preprocessor_function,
        pipeline_name,
        subset=False,
        batch_size=batch_size,
        latent_dim=latent_dim,
        use_balanced_dataset=use_balanced_dataset,
    )

    # Save summary of parameters
    with open(f"results/{pipeline_name}/params.json", "w") as f:
        params = {
            "pipeline_name": pipeline_name,
            "train_data_pickle_fname": train_data_pickle_fname,
            "test_data_pickle_fname": test_data_pickle_fname,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "d_learning_rate": d_learning_rate,
            "g_learning_rate": g_learning_rate,
            "preprocessing_method": str(preprocessor_function),
            "num_classes": num_classes,
            "synthetic_to_real_ratio": synthetic_to_real_ratio,
            "use_balanced_dataset": use_balanced_dataset,
        }
        json.dump(params, f, indent=4)
    logging.info(f"Parameters summary saved in results/{pipeline_name}/params.json")

    # train GAN
    logging.info("Training GAN...")
    gan_history = gan_pipeline.compile_and_fit_GAN(
        d_learning_rate=d_learning_rate,
        g_learning_rate=g_learning_rate,
        beta_1=0.50,
        epochs=num_epochs,
    )
    # plot and save losses
    logging.info("Plotting and saving losses...")
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
    logging.info("Loading test data...")
    X_test, y_test = load_testdata(test_data_pickle_fname, num_classes)
    if debug:
        # & Debug GAN
        logging.debug(X_test.mean(axis=1))
        logging.info("Using test X!!!")
        logging.debug(y_test[:5])
        unraveled_y = y_test.ravel()
        X_test[unraveled_y == "normal"] = 0.2 + np.random.uniform(
            -0.01, 0.01, size=X_test[unraveled_y == "normal"].shape
        )
        X_test[unraveled_y == "attacker"] = 0.8 + np.random.uniform(
            -0.01, 0.01, size=X_test[unraveled_y == "attacker"].shape
        )
        logging.debug("X_test mean:")
        logging.debug(X_test[unraveled_y == "normal"].mean(axis=1).mean())
        logging.debug("y_test mean:")
        logging.debug(X_test[unraveled_y == "attacker"].mean(axis=1).mean())

    # generate and evaluate synthetic data
    logging.info("Evaluating synthetic data...")
    summary_df = generate_and_eval_dataset_once(
        gan_pipeline,
        X_test,
        y_test,
        num_classes,
        synthetic_to_real_ratio=synthetic_to_real_ratio,
    )

    # save summary data
    with open(f"results/{pipeline_name}/metrics{fold}.json", "w") as f:
        json.dump(summary_df, f, indent=4)
    return {"summary_df": summary_df, "gan": gan_pipeline, "losses": losses}
