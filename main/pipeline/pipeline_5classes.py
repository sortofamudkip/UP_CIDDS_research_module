from pathlib import Path
import UP_CIDDS_research_module.main.load_data as loader
import UP_CIDDS_research_module.main.preprocess_data as preprocessor
import UP_CIDDS_research_module.main.score_dataset as dataset_scorer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import UP_CIDDS_research_module.main.gan_models.abstract as AbstractPipeline
import UP_CIDDS_research_module.main.gan_models.basic_gan as basic_gan
import UP_CIDDS_research_module.main.discrim_models.models as models
import UP_CIDDS_research_module.main.score_model as scoring
import UP_CIDDS_research_module.main.synthetic_eval.evaluate_synthetic as tstr_eval
import json
from contextlib import redirect_stdout


def _run_pipeline_5classes(
    pipeline_name: str,  # ex: "N_2_25epochs_TSTR001"
    train_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_train"
    test_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_test"
    num_epochs: int,  # ex: 5
    batch_size: int = 1024,
    learning_rate: float = 0.00001,
):
    ################################
    #                              #
    #   Load data & init pipline   #
    #                              #
    ################################
    basic_gan_pipeline = basic_gan.BasicGANPipeline(
        train_data_pickle_fname,
        preprocessor.decode_N_WGAN_GP,
        pipeline_name,
        subset=False,
        batch_size=batch_size,
    )

    print("Loading training data and initialising GAN...")
    # Save summary of parameters
    with open(f"results/{pipeline_name}/params.json", "w") as f:
        params = {
            "pipeline_name": pipeline_name,
            "train_data_pickle_fname": train_data_pickle_fname,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "preprocessing_method": "N",
            "num_classes": 5,
        }
        json.dump(params, f, indent=4)
    print(f"Parameters summary saved in results/{pipeline_name}/params.json")

    ################################
    #                              #
    #          Train GAN           #
    #                              #
    ################################
    print("Training GAN...")
    gan_history = basic_gan_pipeline.compile_and_fit_GAN(
        learning_rate=learning_rate, beta_1=0.90, epochs=num_epochs
    )

    ################################
    #                              #
    #     Plot and save losses     #
    #                              #
    ################################
    print("Plotting and saving losses...")
    losses = pd.DataFrame(
        {
            "g_loss": gan_history.history["g_loss"],
            "d_loss": gan_history.history["d_loss"],
        }
    ).rename_axis("Epoch")
    losses.to_csv(f"results/{pipeline_name}/losses.csv")
    losses.plot(title="GAN Losses", ylabel="loss").get_figure().savefig(
        f"results/{pipeline_name}/losses.jpg"
    )

    ################################
    #                              #
    #        Load test data        #
    #                              #
    ################################
    print("Loading test data...")
    with open(test_data_pickle_fname, "rb") as f:
        X_test, y_test, y_encoder, X_colnames, X_test_encoders = pickle.load(f)
    y_test = y_encoder.inverse_transform(y_test)

    ################################
    #                              #
    #   Evaluate synthetic data    #
    #                              #
    ################################
    print("Evaluating synthetic data...")
    summary_df = tstr_eval.eval_all_synthetic_5classes(
        basic_gan_pipeline, X_test, y_test, pipeline_name, num_epochs
    )

    summary_df[["plaus_score"]].plot(
        title="Plausibility score, N preprocessing, 5 classes",
        ylabel="Plaus. score",
        xlabel="Epoch",
    ).get_figure().savefig(f"results/{pipeline_name}/plaus.jpg")
    summary_df[["TSTR_tree_f1", "TSTR_perceptron_f1"]].plot(
        title="TSTR F1 scores, N preprocessing, 5 classes",
        ylabel="F1 score",
        xlabel="Epoch",
    ).get_figure().savefig(f"results/{pipeline_name}/f1.jpg")
    summary_df.plot(title="All scores").get_figure().savefig(
        f"results/{pipeline_name}/all_scores.jpg"
    )
    summary_df.to_csv(f"results/{pipeline_name}/scores.csv")
    return summary_df


def run_pipeline_5classes(
    pipeline_name: str,  # ex: "N_2_25epochs_TSTR001"
    train_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_train"
    test_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_test"
    num_epochs: int,  # ex: 5
    batch_size: int = 1024,
    learning_rate: float = 0.00001,
):
    output_dir = Path(__file__).parent / "../../../results" / pipeline_name
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        print(f"Created output dir {output_dir}.")
    except FileExistsError:
        print(f"Dir already exists: {output_dir} ")
        assert False

    with open(output_dir / "log.txt", "w") as f:
        with redirect_stdout(f):
            _run_pipeline_5classes(
                pipeline_name,
                train_data_pickle_fname,
                test_data_pickle_fname,
                num_epochs,
                batch_size,
                learning_rate,
            )
