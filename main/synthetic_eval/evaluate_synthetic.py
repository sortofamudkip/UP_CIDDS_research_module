import UP_CIDDS_research_module.main.preprocess_data as preprocessor
import UP_CIDDS_research_module.main.score_dataset as dataset_scorer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import UP_CIDDS_research_module.main.gan_models.basic_gan as basic_gan
import UP_CIDDS_research_module.main.discrim_models.models as models
import UP_CIDDS_research_module.main.score_model as scoring


def eval_all_synthetic_2classes(
    gan_pipeline: basic_gan.BasicGANPipeline,
    X_test: np.array,
    y_test: np.array,
    pipeline_name: str,
    num_epochs: int,
):
    plaus_scores = []
    tstr_tree_f1s = []
    tstr_perceptron_f1s = []
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} evaluation:\n=========")
        # load created data
        synthetic_samples = np.load(
            f"results/{pipeline_name}/synthetic_epoch{epoch}.npy"
        )
        # evaluate plausibility score
        decoded_samples = gan_pipeline.decode_samples_to_human_format(synthetic_samples)
        plausibility_score = dataset_scorer.score_data_plausibility_single(
            decoded_samples, num_classes=2
        )
        plaus_scores.append(plausibility_score)
        # print(f"overall plausibility_score: {plausibility_score}")
        # evaluate on linear and non-linear models
        X_train, y_train, y_encoder = gan_pipeline.decode_samples_to_model_format(
            synthetic_samples
        )
        if len(np.unique(y_train)) == 1:
            print("y only has one unique label!")
            y_predict_forest = False
            y_predict_logreg = False
            tstr_N_2classes_forest_report = scoring.mode_collapse_binary_stats(
                "TSTR_N_2classes_forest"
            )
            tstr_N_5classes_logreg_report = scoring.mode_collapse_binary_stats(
                "TSTR_N_2classes_logreg"
            )
        else:
            y_predict_forest = models.random_forest_train_predict(
                X_train, X_test, y_train, y_test, y_encoder, n_estimators=20, n_jobs=4
            )
            y_predict_logreg = models.logistic_reg_train_predict(
                X_train,
                X_test,
                y_train,
                y_test,
                y_encoder,
                solver="newton-cg",
                n_jobs=4,
            )
            # assemble reports
            tstr_N_2classes_forest_report = scoring.binary_stats(
                y_test, y_predict_forest, "TSTR_N_2classes_forest", y_encoder
            )
            tstr_N_5classes_logreg_report = scoring.binary_stats(
                y_test, y_predict_logreg, "TSTR_N_2classes_logreg", y_encoder
            )
        tstr_tree_f1s.append(tstr_N_2classes_forest_report["f1_score"][0])
        tstr_perceptron_f1s.append(tstr_N_5classes_logreg_report["f1_score"][0])

    summary_df = pd.DataFrame(
        {
            "plaus_score": plaus_scores,
            "TSTR_tree_f1": tstr_tree_f1s,
            "TSTR_perceptron_f1": tstr_perceptron_f1s,
        }
    )
    summary_df.index += 1
    return summary_df


def eval_all_synthetic_5classes(
    gan_pipeline: basic_gan.BasicGANPipeline,
    X_test: np.array,
    y_test: np.array,
    pipeline_name: str,
    num_epochs: int,
):
    plaus_scores = []
    tstr_tree_f1s = []
    tstr_perceptron_f1s = []
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} evaluation:\n=========")
        # load created data
        synthetic_samples = np.load(
            f"results/{pipeline_name}/synthetic_epoch{epoch}.npy"
        )
        # evaluate plausibility score
        decoded_samples = gan_pipeline.decode_samples_to_human_format(synthetic_samples)
        plausibility_score = dataset_scorer.score_data_plausibility_single(
            decoded_samples, num_classes=5
        )
        plaus_scores.append(plausibility_score)
        # print(f"overall plausibility_score: {plausibility_score}")
        # evaluate on linear and non-linear models
        X_train, y_train, y_encoder = gan_pipeline.decode_samples_to_model_format(
            synthetic_samples
        )
        if len(np.unique(y_train)) == 1:
            print("y only has one unique label!")
            y_predict_forest = False
            y_predict_logreg = False
            tstr_N_5classes_forestF1 = 0
            tstr_N_5classes_logregF1 = 0
        else:
            y_predict_forest = models.random_forest_train_predict(
                X_train, X_test, y_train, y_test, y_encoder, n_estimators=20, n_jobs=4
            )
            y_predict_logreg = models.logistic_reg_train_predict(
                X_train,
                X_test,
                y_train,
                y_test,
                y_encoder,
                solver="newton-cg",
                n_jobs=4,
            )
            # assemble reports
            tstr_N_5classes_forestF1 = scoring.multiclass_f1score_weighted(
                y_test, y_predict_forest
            )
            tstr_N_5classes_logregF1 = scoring.multiclass_f1score_weighted(
                y_test, y_predict_logreg
            )
        tstr_tree_f1s.append(tstr_N_5classes_forestF1)
        tstr_perceptron_f1s.append(tstr_N_5classes_logregF1)

    summary_df = pd.DataFrame(
        {
            "plaus_score": plaus_scores,
            "TSTR_tree_f1": tstr_tree_f1s,
            "TSTR_perceptron_f1": tstr_perceptron_f1s,
        }
    )
    summary_df.index += 1
    return summary_df
