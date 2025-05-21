# +
import logging
from pathlib import Path
from pickle import load

import pandas as pd

from ..utils.config_reader import read_config
from ..utils.training_evaluation import (decile_report, percentile_report,
                                         save_metrics)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score, r2_score


# -


if __name__ == "__main__":

    PROJECT_PATH = Path(__file__).parents[1]
    DATA_PATH = PROJECT_PATH / "data"
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

    logging.info("Loading test data...")
    if test_file.endswith(".txt"):
        test = pd.read_csv(
            DATA_PATH / test_file, sep=";", decimal=",", dtype="object", encoding="latin1"
        )  # ,nrows=10000
    else:
        if test_file == "":
            test_file = f"original_testing_data_{version.replace('.','-')}.csv"
        test = pd.read_csv(DATA_PATH / test_file, dtype="object", encoding="latin1")

    special_columns = [config.id_column, "ANO_MES"]

    X_special = test[special_columns].copy()

    # USE THE SAME PROCESSORS THAT WERE APPLYED WHEN CREATING THE TRAINING DATA
    logging.info("Loading and applying first processor...\n")
    processor_1 = load(open(MODEL_PATH / "preprocess" /
                            f"processor1_{version.replace('.', '-')}.pkl", "rb"))


    test = processor_1.transform(test)
    del processor_1
    X_special = test[special_columns].copy()
    if "TARGET" in test.columns:
        X_test = test.drop(columns=special_columns + ["TARGET"])
        y_test = test["TARGET"]
    else:
        y_test = None
    del test
    logging.info("Loading and applying second processor...\n")
    processor_2 = load(open(MODEL_PATH / "preprocess" /
                            f"processor2_{version.replace('.', '-')}.pkl", "rb"))
    X_test = processor_2.transform(X_test)
    del processor_2
    logging.info("Loading and applying third processor...\n")
    processor_3 = load(open(MODEL_PATH / "preprocess" /
                            f"processor3_{version.replace('.', '-')}.pkl", "rb"))
    X_test = processor_3.transform(X_test)
    del processor_3
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]

    # END OF TRANSFORMING THE TEST DATA
    logging.info(f"Saving transformed dataset")
    transformed = pd.concat([X_special, X_test, y_test], axis=1)
    transformed.to_csv(DATA_PATH / f"transformed_test_dataset_{version.replace('.', '-')}.csv", 
                                           index=False, encoding="utf-8")

    # METRICS FOR TRAIN
    logging.info("Loading model and generating predictions for train...\n")
    train_df = pd.read_csv(
        DATA_PATH / f"training_data_{version.replace('.','-')}.csv", encoding="latin1", dtype={config.id_column:'object'}
    )
    X_train = train_df.drop(columns=special_columns + ["TARGET"])
    y_train = train_df["TARGET"]

    predictor = load(open(MODEL_PATH / "models" / f"training_pipeline_{version.replace('.','-')}.pkl", "rb"))

    y_pred = predictor.predict_proba(X_train)
    
    if len(y_pred) > 100 and y_train is not None:
        #         # Data to plot precision - recall curve
        #         precision, recall, thresholds = precision_recall_curve(y_test, y_pred[:,1])
        #         # Use AUC function to calculate the area under the curve of precision recall curve
        #         auc_precision_recall = auc(recall1, precision1)
        #         print("Area under the curve Precision-Recall:", auc_precision_recall)
        #         logging.info("Area under the curve ROC: %f", roc_auc_score(y_test, y_pred[:, 1]))
    
        decile_report(y_train, y_pred[:, 0], output="print")
        #percentile_report(y_train, y_pred[:, 0], output="print")

        roc_score = roc_auc_score(y_train, 1 - y_pred[:, 0])
        logging.info("Area under the curve ROC: %f", roc_score)
        #f1_score = f1_score(y_train, 1 - y_pred[:, 0])
        #logging.info("F1 score: %f", f1_score)
        #R2 = r2_score(y_train, 1 - y_pred[:, 0])
        #logging.info("R2 score: %f", R2)
    
    else:
        for client_id, pred in zip(X_special[config.id_column], y_pred[:, 0]):
            logging.info("ID: %s, Prediction: %f", client_id, pred)

    # METRICS FOR TEST
    logging.info("Loading model and generating predictions test...\n")

    y_pred = predictor.predict_proba(X_test)

    if len(y_pred) > 100 and y_test is not None:
        #         # Data to plot precision - recall curve
        #         precision, recall, thresholds = precision_recall_curve(y_test, y_pred[:,1])
        #         # Use AUC function to calculate the area under the curve of precision recall curve
        #         auc_precision_recall = auc(recall1, precision1)
        #         print("Area under the curve Precision-Recall:", auc_precision_recall)
        #         logging.info("Area under the curve ROC: %f", roc_auc_score(y_test, y_pred[:, 1]))

        decile_report(y_test, y_pred[:, 0], output="print")
        #percentile_report(y_test, y_pred[:, 0], output="print")

        save_metrics(y_test, y_pred[:, 0], test_file.split(".")[0], REPORT_PATH, version)

        from pytimedinput import timedInput

        model_comparison, timedOut = timedInput(
            "Do you want to compare the results with other model?[y/n]", timeout=1 * 60
        )  # timeout in seconds
        if (model_comparison.lower() == "y" or model_comparison == "") and not timedOut:
            old_version = input("Please indicate the version that you want to compare with (e.g. v00.0):")
            old_predictor = load(
                open(MODEL_PATH / "models" / f"training_pipeline_{old_version.replace('.','-')}.pkl", "rb")
            )
            y_pred_old = old_predictor.predict_proba(X_test)
            logging.info("Area under the curve ROC: %f", roc_auc_score(y_test, y_pred_old[:, 1]))

            decile_report(y_test, y_pred_old[:, 0], output="print")
            #percentile_report(y_test, y_pred_old[:, 0], output="print")

            save_metrics(y_test, y_pred_old[:, 0], test_file.split(".")[0], REPORT_PATH, old_version)
    else:
        for client_id, pred in zip(X_special[config.id_column], y_pred[:, 0]):
            logging.info("ID: %s, Prediction: %f", client_id, pred)
