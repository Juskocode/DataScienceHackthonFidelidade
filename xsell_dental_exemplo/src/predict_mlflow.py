import logging
from pathlib import Path

import mlflow
import pandas as pd
from azureml.core import Workspace
# from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from ..utils.config_reader import read_config
from ..utils.training_evaluation import (decile_report, percentile_report,
                                         save_metrics)

# Set the tracking URI to your AzureML workspace
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

if __name__ == "__main__":

    PROJECT_PATH = Path(__file__).parents[1]
    DATA_PATH = PROJECT_PATH / "data"
    FEATURES_PATH = PROJECT_PATH / "features"
    MODEL_PATH = PROJECT_PATH / "models"
    REPORT_PATH = PROJECT_PATH / "report"
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

    config = read_config(CONFIG_PATH / "predict_config.yml")

    # Access the values of the arguments
    test_file = config.input_file
    version = config.version

    model_name = f"{PROJECT_PATH.name}_{version.replace('.','-')}"
    model_version = "latest"

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

    logging.info("Loading test data...")
    if test_file.endswith(".txt"):
        test = pd.read_csv(
            DATA_PATH / test_file, sep=";", decimal=",", dtype="object", encoding="latin1"
        )  # ,nrows=10000
    else:
        if test_file == "":
            test_file = f"original_testing_data_{version.replace('.','-')}.csv"
        test = pd.read_csv(DATA_PATH / test_file, dtype="object", encoding="latin1")

    input_tuple = (test, config.id_column)
    predictions = model.predict(input_tuple)

    if predictions["TARGET"].isna().all() or len(predictions) <= 100:
        for client_id, pred in zip(predictions[config.id_column], predictions["Prediction_0"]):
            logging.info("ID: %s, Prediction: %f", client_id, pred)

    else:
        #         logging.info("Area under the curve ROC: %f", roc_auc_score(predictions["TARGET"], predictions["Prediction_1"]))

        decile_report(predictions["TARGET"], predictions["Prediction_0"], output="print")
        percentile_report(predictions["TARGET"], predictions["Prediction_0"], output="print")

        save_metrics(
            predictions["TARGET"], predictions["Prediction_0"], test_file.split(".")[0], REPORT_PATH, version
        )

        from pytimedinput import timedInput

        model_comparison, timedOut = timedInput(
            "Do you want to compare the results with other model?[y/n]", timeout=1 * 60
        )  # timeout in seconds
        if (model_comparison.lower() == "y" or model_comparison == "") and not timedOut:
            old_version = input("Please indicate the version that you want to compare with (e.g. v00.0):")

            old_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{old_version}")
            old_predictions = old_model.predict(test, config.id_column)
            logging.info(
                "Area under the curve ROC: %f",
                roc_auc_score(old_predictions["TARGET"], old_predictions["Prediction_1"]),
            )

            decile_report(old_predictions["TARGET"], old_predictions["Prediction_0"], output="print")
            percentile_report(old_predictions["TARGET"], old_predictions["Prediction_0"], output="print")

            save_metrics(
                old_predictions["TARGET"],
                old_predictions["Prediction_0"],
                test_file.split(".")[0],
                REPORT_PATH,
                old_version,
            )
