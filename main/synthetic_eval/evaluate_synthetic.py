import UP_CIDDS_research_module.main.preprocess_data as preprocessor
import UP_CIDDS_research_module.main.score_dataset as dataset_scorer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import UP_CIDDS_research_module.main.gan_models.basic_gan as basic_gan
import UP_CIDDS_research_module.main.discrim_models.models as models
import UP_CIDDS_research_module.main.score_model as scoring


def eval_synthetic_one_epoch(
    gan_pipeline: basic_gan.BasicGANPipeline,
    synthetic_samples: np.array,
    X_test: np.array,
    y_test: np.array,
    num_classes: int,
):
    print("Begin synthetic evaluation")
    plausibility_score = eval_plaus_score(gan_pipeline, synthetic_samples)
    # evaluate on linear and non-linear models
    X_train, y_train, y_encoder = decode_samples_to_np(gan_pipeline, synthetic_samples)
    if len(np.unique(y_train)) == 1:
        print("y only has one unique label!")
        tstr_forest_f1, tstr_logreg_f1 = 0, 0
        tstr_forest_roc_auc, tstr_logreg_roc_auc = 0, 0
    else:
        # train models and obtain predictions
        (
            y_predict_and_proba_forest,
            y_predict_and_proba_logreg,
        ) = tstr_predict_forest_logreg(
            X_test, y_test, X_train, y_train, y_encoder, also_return_proba=True
        )
        y_predict_forest, y_proba_forest = y_predict_and_proba_forest
        y_predict_logreg, y_proba_logreg = y_predict_and_proba_logreg
        # evaluate models
        ## forest (i.e. nonlinear)
        tstr_forest_f1 = scoring.f1_stats_one_epoch(
            y_test, y_predict_forest, num_classes
        )
        tstr_forest_roc_auc = (
            scoring.roc_auc_one_epoch(y_test, y_proba_forest[:, 1])
            if num_classes == 2
            else -1
        )
        ## logreg (i.e. linear)
        tstr_logreg_f1 = scoring.f1_stats_one_epoch(
            y_test, y_predict_logreg, num_classes
        )
        tstr_logreg_roc_auc = (
            scoring.roc_auc_one_epoch(y_test, y_proba_logreg[:, 1])
            if num_classes == 2
            else -1
        )
    # gather results
    summary_df = {
        "plaus_score": plausibility_score,
        "TSTR_forest_f1": tstr_forest_f1,
        "TSTR_forest_roc_auc": tstr_forest_roc_auc,
        "TSTR_logreg_f1": tstr_logreg_f1,
        "TSTR_logreg_roc_auc": tstr_logreg_roc_auc,
    }
    return summary_df


def tstr_predict_forest_logreg(
    X_test, y_test, X_train, y_train, y_encoder, also_return_proba=False
):
    y_predict_forest = models.random_forest_train_predict(
        X_train,
        X_test,
        y_train,
        y_test,
        y_encoder,
        also_return_proba=also_return_proba,
        n_estimators=20,
        n_jobs=4,
    )
    # print(f"y_predict_forest: {y_predict_forest}")
    y_predict_logreg = models.logistic_reg_train_predict(
        X_train,
        X_test,
        y_train,
        y_test,
        y_encoder,
        also_return_proba=also_return_proba,
        solver="newton-cg",
        n_jobs=4,
    )

    return y_predict_forest, y_predict_logreg


def eval_all_synthetic_2classes(
    gan_pipeline: basic_gan.BasicGANPipeline,
    X_test: np.array,
    y_test: np.array,
    pipeline_name: str,
    num_epochs: int,
):
    plaus_scores = []
    tstr_forest_f1s = []
    tstr_logreg_f1s = []
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} evaluation:\n=========")
        # load created data
        synthetic_fname = f"results/{pipeline_name}/synthetic_epoch{epoch}.npy"
        synthetic_samples = load_synthetic_data(synthetic_fname)
        # evaluate plausibility score
        plausibility_score = eval_plaus_score(gan_pipeline, synthetic_samples)
        plaus_scores.append(plausibility_score)
        # evaluate on linear and non-linear models
        X_train, y_train, y_encoder = decode_samples_to_np(
            gan_pipeline, synthetic_samples
        )
        if len(np.unique(y_train)) == 1:
            print("y only has one unique label!")
            tstr_N_2classes_forest_report = scoring.mode_collapse_binary_stats(
                "TSTR_2classes_forest"
            )
            tstr_N_5classes_logreg_report = scoring.mode_collapse_binary_stats(
                "TSTR_2classes_logreg"
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
        tstr_forest_f1s.append(tstr_N_2classes_forest_report["f1_score"][0])
        tstr_logreg_f1s.append(tstr_N_5classes_logreg_report["f1_score"][0])

    summary_df = pd.DataFrame(
        {
            "plaus_score": plaus_scores,
            "TSTR_tree_f1": tstr_forest_f1s,
            "TSTR_perceptron_f1": tstr_logreg_f1s,
        }
    )
    summary_df.index += 1
    return summary_df


def decode_samples_to_np(gan_pipeline, synthetic_samples):
    return gan_pipeline.decode_samples_to_model_format(synthetic_samples)


def load_synthetic_data(synthetic_fname: str):
    return np.load(synthetic_fname)


def eval_plaus_score(
    gan_pipeline: basic_gan.BasicGANPipeline, synthetic_samples: np.array
):
    decoded_samples = gan_pipeline.decode_samples_to_human_format(synthetic_samples)
    plausibility_score = dataset_scorer.score_data_plausibility_single(
        decoded_samples, num_classes=2
    )

    return plausibility_score


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
