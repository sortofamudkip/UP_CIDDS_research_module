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


def create_output_dir(pipeline_name: str, skip_if_exists=False) -> Path:
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
):
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
