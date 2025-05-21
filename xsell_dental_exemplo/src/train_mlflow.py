import json
import logging
from itertools import product
from pathlib import Path
from pickle import dump, load
from time import gmtime, strftime

import mlflow
import numpy as np
import pandas as pd
# +
from mlflow.tracking import MlflowClient
#from azureml.core import Workspace
from pytimedinput import timedInput
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils import data_processing as dp, feature_mapping
from ..utils import select_features as feat_sel
from ..utils import feature_mapping as feat_map
from ..utils import training_evaluation as tr
from ..utils import model_wrapper as mw
from ..utils.config_reader import read_config

# Set the tracking URI to your AzureML workspace
#ws = Workspace.from_config()
#mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


# -


# # Serialize an in-memory pipeline and log it as an artifact
# def log_in_memory_artifact(artifact, artifact_path):
#     import os
#     import tempfile
#     from io import BytesIO

#     buffer = BytesIO()
#     dump(artifact, buffer)
#     buffer.seek(0)

#     # Create a temporary file
#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(buffer.getvalue())
#         temp_file_path = temp_file.name

#     # Log the temporary file as an artifact
#     mlflow.log_artifact(temp_file_path, artifact_path)

#     # Clean up the temporary file
#     os.remove(temp_file_path)


if __name__ == "__main__":

    PROJECT_PATH = Path(__file__).parents[1]
    # PROJECT_PATH = Path(globals()['_dh'][0]).parent
    DATA_PATH = PROJECT_PATH / "data"
    MODEL_PATH = PROJECT_PATH / "models"
    REPORT_PATH = PROJECT_PATH / "report"
    AUX_REPORT_PATH = PROJECT_PATH / "report" / "Auxiliary Reports"
    CONFIG_PATH = PROJECT_PATH / "configs"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(PROJECT_PATH / "src" / "logfile.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    config = read_config(CONFIG_PATH / "training_config.yml")

    version = config.version
    old_version = config.old_version
    train_type = config.train_type
    fixed_features = config.fixed_features if config.fixed_features is not None else []

    mlflow.set_experiment(experiment_name=PROJECT_PATH.name)
    #     mlflow.autolog()
    with mlflow.start_run() as run:
        mlflow.log_params(
            {k: v for k, v in config.get_params().items() if k != "hyperparameters"}
        )  # Excluding hyperparameters because of the size limit

        logging.info(f"Loading training data: training_data_{version.replace('.','-')}.csv...")
        train_df = pd.read_csv(
            DATA_PATH / f"training_data_{version.replace('.','-')}.csv", encoding="latin1", dtype={config.id_column:'object'}
        )
        train_df = train_df[~train_df["TARGET"].isna()].copy()
        special_columns = [config.id_column, "ANO_MES"]

        

        cur_pipeline = {}
        best_pipeline = {}
        best_score = 0
        best_model_id = ""
        train_str = "Train_"
        flag_auto_features = "automatic" in config.n_features

        if train_type != "recalibration":
            logging.info("Start training...")
            for m in config.months:
                training_grid = config.get_training_transformer_grid()
                combinations = product(*training_grid.values())

                X_special = tr.filter_date(train_df, m)[special_columns]
                X_train = tr.filter_date(train_df, m).drop(columns=special_columns + ["TARGET"])
                y_train = tr.filter_date(train_df, m)["TARGET"]

                tt = dp.TrainingTransformer(version=version, folder=AUX_REPORT_PATH)
                # Iterate over the combinations
                for combo in combinations:
                    cur_grid = dict(zip(training_grid.keys(), combo))
                    logging.info("")
                    logging.info(cur_grid)

                    if cur_grid["binning"] is True and cur_grid["outlier_cleaning"] is True:
                        # We do nothing when both are True
                        logging.info(
                            "Binning and Outlier_cleaning are both True \
                        so nothing will be done because we do one or the other."
                        )
                        continue
                    else:
                        tt.set_grid(cur_grid)
                        X_train = tt.run_ColumnsAggregator(X_train, y_train, len(training_grid["column_aggregation"]))
                        X_train = tt.run_FlagCreation(X_train, y_train, len(training_grid["flag_creation"]))
                        X_train = tt.run_BinningTransformer(X_train, y_train, len(training_grid["binning"]))
                        X_train = tt.run_OutlierTransformer(X_train, y_train, len(training_grid["outlier_cleaning"]))
                        X_train = tt.run_ValueCountsFilter(X_train, y_train)
                        cur_pipeline = tt.output_pipeline()

                        selected_features = feat_sel.make_feature_selection(
                            X_train,
                            y_train,
                            version,
                            AUX_REPORT_PATH,
                            train_str + str(m) + "Months_" + tt.output_string(),
                            config.overwrite_saved_features,
                            display=False,
                        )
                        # creates a list with the fixed features + the features the came from the feature selection in the same order
                        selected_features = pd.Series(fixed_features + selected_features).drop_duplicates().tolist()

                        if flag_auto_features:
                            # Genereate number of features based on a geometric series
                            config.n_features = np.geomspace(10, len(selected_features), num=8, dtype=int)[:3]
                        for v in config.n_features:
                            # Define preprocessing steps
                            cur_pipeline["ColumnSelector"] = ColumnTransformer(
                                transformers=[
                                    ("selected", "passthrough", selected_features[: int(v)])
                                ],  # 'passthrough' to keep selected columns unchanged
                                remainder="drop",
                            ).set_output(
                                transform="pandas"
                            )  # Drop columns not specified in 'selected_features'
                            X_train_sel_cols = cur_pipeline["ColumnSelector"].fit_transform(X_train)

                            logging.info(
                                "start model training: %s",
                                train_str + str(m) + "Months_" + tt.output_string() + "_" + str(v),
                            )
                            if config.standard_scaler:
                                cur_pipeline["StandardScaler"] = StandardScaler().set_output(transform="pandas")
                                X_train_sel_cols = cur_pipeline["StandardScaler"].fit_transform(X_train_sel_cols)
                            else:
                                cur_pipeline["StandardScaler"] = None
                            alg = tr.model_training(
                                X_train_sel_cols, y_train, config.hyperparameters, roc_threshold=0.5
                            )
                            for model in alg:
                                mlflow.log_metric(
                                    f"ROC_AUC_{train_str + str(m)}Months_{tt.output_string()}_{str(v)}_{type(model[0].estimator).__name__}",
                                    model[1],
                                )
                                # log_in_memory_artifact(model[0], artifact_path=f"model_{train_str + str(m)}Months_{tt.output_string()}_{str(v)}_{type(model[0].estimator).__name__}")
                                if model[1] > best_score:
                                    best_model_id = train_str + str(m) + "Months_" + tt.output_string() + "_" + str(v)
                                    logging.info(
                                        "current best model: %s - %s, Score: %s",
                                        best_model_id,
                                        model[0],
                                        model[1],
                                    )
                                    best_pipeline = {k: i for k, i in cur_pipeline.items()}
                                    best_pipeline["Model"] = model[0]
                                    best_score = model[1]

        if train_type != "train":
            logging.info("Start recalibration...")
            predictor = load(
                open(MODEL_PATH / "models" / f"training_pipeline_{old_version.replace('.','-')}.pkl", "rb")
            )

            for m in config.months:
                X_special = tr.filter_date(train_df, m)[special_columns]
                X_train = tr.filter_date(train_df, m).drop(columns=special_columns + ["TARGET"])
                y_train = tr.filter_date(train_df, m)["TARGET"]
                predictor.fit(X_train, y_train)
                score = cross_val_score(predictor, X_train, y_train, cv=3, scoring="roc_auc").mean()
                logging.info(f"recalibration model with {m} months finished with a score: {score}")
                mlflow.log_metric(f"ROC_AUC_{train_str + str(m)}Months_recalibration", score)
                if score > best_score:
                    best_model_id = train_str + str(m) + "Months" + "_recalibration"
                    logging.info(
                        "%s - current best model: %s, Score: %s",
                        strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                        best_model_id,
                        score,
                    )
                    best_score = score
                    best_pipeline = predictor

        if type(best_pipeline) is dict:
            pipe = Pipeline([(n, t) for n, t in best_pipeline.items() if t is not None])
        else:
            pipe = best_pipeline
            best_pipeline = pipe.named_steps
            for i in ["ColumnsAggregator", "FlagCreation", "Binning", "OutlierTransformer", "StandardScaler"]:
                if i not in best_pipeline.keys():
                    best_pipeline[i] = None

        logging.info("Final model: %s", pipe)
        if best_pipeline == {}:
            logging.warning("There is no valid model for selection.")
        else:
            dump(pipe, open(MODEL_PATH / "models" / f"training_pipeline_{version.replace('.','-')}.pkl", "wb"))

            filename = "best_models.json"
            with open(AUX_REPORT_PATH / filename) as file:
                best_models = json.load(file)

            best_models[version] = [
                {
                    "ColumnsAggregator": best_pipeline["ColumnsAggregator"] is not None,
                    "FlagCreation": best_pipeline["FlagCreation"] is not None,
                    "Binning": best_pipeline["Binning"] is not None,
                    "OutlierTransformer": best_pipeline["OutlierTransformer"] is not None,
                    "ValueCountsFilter": best_pipeline["ValueCountsFilter"] is not None,
                    "ColumnSelector": best_pipeline["ColumnSelector"].transformers_[0][-1],
                    "StandardScaler": best_pipeline["StandardScaler"] is not None,
                    "Model": str(best_pipeline["Model"].get_params()["estimator"])
                    .replace("\t", "")
                    .replace("\n", "")
                    .replace(" ", ""),
                }
            ]
            with open(AUX_REPORT_PATH / filename, "w") as file:
                json.dump(best_models, file, ensure_ascii=False)

            with open(AUX_REPORT_PATH / "selected_features.json") as file:
                selected_features = json.load(file)
            selected_features[version] = best_models[version][0]["ColumnSelector"]
            with open(AUX_REPORT_PATH / "selected_features.json", "w") as file:
                json.dump(selected_features, file, ensure_ascii=False)

            for t in best_pipeline.values():
                if hasattr(t, "feature_mapping"):
                    feature_mapping[version].update(t.feature_mapping)

            feature_mapping[version] = feat_map.clean_feature_mapping(feature_mapping[version])

            with open(AUX_REPORT_PATH / "feature_mapping.json", "w") as file:
                json.dump(feature_mapping, file, ensure_ascii=False)

            model_path = f"{PROJECT_PATH.name}_{version.replace('.','-')}"
            mlflow.log_text(
                best_model_id + " - " + repr(best_pipeline["Model"].estimator), model_path + "/metadata/model_id.txt"
            )

            # log_in_memory_artifact(pipe, artifact_path="best_model")
            mlflow.pyfunc.log_model(
                model_path,
                python_model=mw.ModelWrapper(),
                artifacts={
                    ## ADD YOUR PROCESSORS HERE
                    # "first_processor": str(
                    #     MODEL_PATH / "preprocess" / f"first_processor_{version.replace('.','-')}.pkl"
                    # ),
                    # "second_processor": str(
                    #     MODEL_PATH / "preprocess" / f"second_processor_{version.replace('.','-')}.pkl"
                    # ),
                    "model": str(MODEL_PATH / "models" / f"training_pipeline_{version.replace('.','-')}.pkl"),
                },
                code_paths=[str(PROJECT_PATH / "utils")],
            )

            feature_importances, timedOut = timedInput(
                "Do you want to perform feature importances? It might take a while if the training dataset is big. [y/n]",
                timeout=5 * 60,
            )  # timeout in seconds
            if feature_importances.lower() == "y" or feature_importances == "" or timedOut:
                logging.info("Starting feature importances...")
                n_month = int(best_model_id.split("Months")[0].split("_")[-1])
                X_train = tr.filter_date(train_df, n_month).drop(columns=special_columns + ["TARGET"])
                y_train = tr.filter_date(train_df, n_month)["TARGET"]

                mlflow.log_text(
                    str(selected_features), model_path + "/metadata/selected_features.txt"
                )

                ft_mapping_selected = {
                    k: feature_mapping[version][k]
                    for k in best_models[version][0]["ColumnSelector"]
                    if k in feature_mapping[version].keys()
                }
                mlflow.log_text(str(ft_mapping_selected), model_path + "/metadata/feature_mapping.txt")
                y_pred = pipe.predict_proba(X_train)
                dr = tr.decile_report(y_train, y_pred[:, 0], output="return")
                from sklearn.metrics import roc_auc_score

                roc_treino = round(roc_auc_score(y_train, y_pred[:, 1]), 3)
                lift_cumulativo = dr[dr["Decile"] == 1]["Cumulative Lift"].values[0]
                capturados_cumulativa = dr[dr["Decile"] == 3]["Cumulative rate (%)"].values[0]
                nd = "ND"
                import textwrap

                table_str = textwrap.dedent(
                    f"""
                +---------------------------------------------+
                |                      |  Treino  | Validação |
                +---------------------------------------------+
                | ROC                  |{roc_treino:^10}|{round(best_score,3):^11}|
                +---------------------------------------------+
                | Lift Cumulativo      |{lift_cumulativo:^10}|{nd:^11}|
                | (1º decil)           |          |           |
                +---------------------------------------------+
                | % Capturados         |{capturados_cumulativa:^10}|{nd:^11}|
                | Cumulativa (3º decil)|          |           |
                +---------------------------------------------+
                """
                )
                mlflow.log_text(table_str, model_path + "/metadata/metrics_summary.txt")

                fig1, fig2 = tr.save_feature_importances(pipe, X_train, y_train, REPORT_PATH, version)
                mlflow.log_figure(fig1, model_path + "/reports/Impact of each feature value.png")
                #                 mlflow.log_figure(fig2, model_path+"/reports/Impact of the mean values.png")
                mlflow.log_figure(fig2, model_path + "/reports/Feature Importances.png")

        logging.info("Job completed!")
