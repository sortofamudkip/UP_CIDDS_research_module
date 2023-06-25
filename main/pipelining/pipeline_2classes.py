import UP_CIDDS_research_module.main.preprocess_data as preprocessor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import UP_CIDDS_research_module.main.synthetic_eval.evaluate_synthetic as tstr_eval
import json


def run_pipeline_2classes(
    pipeline_name: str,  # ex: "N_2_25epochs_TSTR001"
    train_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_train"
    test_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_test"
    num_epochs: int,  # ex: 5
    gan_pipeline_class,  # ex: basic_gan.BasicGANPipeline
    preprocessor_func,  # ex: preprocessor.decode_N_WGAN_GP
    batch_size: int = 1024,
    learning_rate: float = 0.00001,
):
    ################################
    #                              #
    #   Load data & init pipline   #
    #                              #
    ################################
    gan_pipeline = gan_pipeline_class(
        train_data_pickle_fname,
        preprocessor_func,
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
            "num_classes": 2,
        }
        json.dump(params, f, indent=4)
    print(f"Parameters summary saved in results/{pipeline_name}/params.json")

    ################################
    #                              #
    #          Train GAN           #
    #                              #
    ################################
    print("Training GAN...")
    gan_history = gan_pipeline.compile_and_fit_GAN(
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
    X_test, y_test = load_testdata_2classes(test_data_pickle_fname)

    ################################
    #                              #
    #   Evaluate synthetic data    #
    #                              #
    ################################
    print("Evaluating synthetic data...")
    summary_df = eval_2classes(pipeline_name, num_epochs, gan_pipeline, X_test, y_test)
    return summary_df


def load_testdata_2classes(test_data_pickle_fname: str):
    with open(test_data_pickle_fname, "rb") as f:
        X_test, y_test, y_encoder, X_colnames, X_test_encoders = pickle.load(f)
    y_test = y_test.ravel()
    y_test = y_encoder.inverse_transform(y_test)
    return X_test, y_test


def eval_2classes(
    pipeline_name: str,
    num_epochs: int,
    gan_pipeline,
    X_test: np.array,
    y_test: np.array,
):
    summary_df = tstr_eval.eval_all_synthetic_2classes(
        gan_pipeline, X_test, y_test, pipeline_name, num_epochs
    )

    summary_df[["plaus_score"]].plot(
        title="Plausibility score, 2 classes",
        ylabel="Plaus. score",
        xlabel="Epoch",
    ).get_figure().savefig(f"results/{pipeline_name}/plaus.jpg")
    summary_df[["TSTR_tree_f1", "TSTR_perceptron_f1"]].plot(
        title="TSTR F1 scores, 2 classes",
        ylabel="F1 score",
        xlabel="Epoch",
    ).get_figure().savefig(f"results/{pipeline_name}/f1.jpg")
    summary_df.plot(title="All scores").get_figure().savefig(
        f"results/{pipeline_name}/all_scores.jpg"
    )
    summary_df.to_csv(f"results/{pipeline_name}/scores.csv")
    return summary_df


# def run_pipeline_2classes(
#     pipeline_name: str,  # ex: "N_2_25epochs_TSTR001"
#     train_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_train"
#     test_data_pickle_fname: str,  # ex: "./preprocessed/X_y_2_classes_N_test"
#     num_epochs: int,  # ex: 5
#     batch_size: int = 1024,
#     learning_rate: float = 0.00001,
# ):
#     output_dir = Path(__file__).parent / "../../../results" / pipeline_name
#     try:
#         output_dir.mkdir(parents=True, exist_ok=False)
#         print(f"Created output dir {output_dir}.")
#     except FileExistsError:
#         print(f"Dir already exists: {output_dir} ")
#         assert False

#     with open(output_dir / "log.txt", "w") as f:
#         with redirect_stdout(f):
#             _run_pipeline_2classes(
#                 pipeline_name,
#                 train_data_pickle_fname,
#                 test_data_pickle_fname,
#                 num_epochs,
#                 batch_size,
#                 learning_rate,
#             )
