import logging
from pathlib import Path
from pickle import load

import pandas as pd
import ast
import numpy as np
from datetime import datetime, timedelta

from ..utils.config_reader import read_config
from ..utils import feature_mapping as feat_map

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

    config = read_config(CONFIG_PATH / "scores_monitoring_config.yml")

    # Access the values of the arguments
    mes_scores = datetime.strftime(config.mes_scores,"%Y%m")
    version = config.version
    id_column = config.id_column
    MODEL_TYPE = config.MODEL_TYPE
    MODEL_VERSION = version[1:].split('.')[0] #getting the integer of the verion
    MODEL_LOB = config.MODEL_LOB
    input_file = f"ELEG_{MODEL_TYPE}_{MODEL_LOB}_V{MODEL_VERSION}_{mes_scores}.csv.gz"

    ## ADD MORE COLUMNS IF NEEDED
    special_columns = ['MODEL_TYPE', 'MODEL_VERSION', 'MODEL_LOB']

    test = pd.read_csv(DATA_PATH / input_file, encoding='latin-1', dtype="object", compression='gzip')
    logging.info(f"data loaded with shape: {test.shape}" )

    test['MODEL_TYPE'] = MODEL_TYPE
    test['MODEL_VERSION'] = MODEL_VERSION
    test['MODEL_LOB'] = MODEL_LOB

    ## INSERT HERE YOUR PROCESSORS TO TRANSFORM THE DATA
    #Note: don't forget to rename your ID column if needed

    # If needed do this:
    X_special = test[special_columns]
    #we should drop the special_columns because they were not used as inputs
    if 'TARGET' in test.columns:
        X_test = test.drop(columns=['TARGET']+special_columns)
        y_test = test['TARGET']
    else:
        X_test = test.drop(columns=special_columns)
        y_test = None

    ## END OF SECTION TO TRANSFORM THE DATA

    # Scores
    logging.info("Loading model and generating predictions...\n")
    predictor = load(open(MODEL_PATH / "models" / f"training_pipeline_{version.replace('.','-')}.pkl", 'rb'))


    # calculate predictions & probabilities
    y_pred = predictor.predict(X_test)
    y_prob = predictor.predict_proba(X_test)[:,1]
    print("predict and probabilities calculated")


    # Reshape to column vectors
    y_pred = y_pred.reshape(-1, 1)
    y_prob = y_prob.reshape(-1, 1)

    # Horizontally stack the arrays
    result = np.hstack([y_pred, y_prob])

    # Create a new DataFrame with the concatenated results
    df_predictions = pd.DataFrame(result, columns=['prediction', 'EM_EVENTPROBABILITY'])

    # create percentiles
    df_predictions = df_predictions.sort_values('EM_EVENTPROBABILITY',ascending=False)
    df_predictions['i'] = range(1, len(X_special) + 1)
    # Use the pandas.qcut function to create ranks
    df_predictions['rank_i'] = pd.qcut(df_predictions['i'], q=100, labels=False, duplicates='drop', retbins=False)
    df_predictions['PERCENTILE'] = df_predictions['rank_i'] + 1


    # get final dataset only with columns of interest
    final_dataset = pd.concat([X_special, df_predictions], axis=1)

    cols_interest = ['ANO_MES', id_column]+special_columns + [
    'EM_EVENTPROBABILITY',
    'PERCENTILE'
    ]

    final_dataset = final_dataset[cols_interest].sort_values('EM_EVENTPROBABILITY', ascending = False)

    # Save dataset
    result_file = f"SC_{MODEL_TYPE}_{MODEL_LOB}_V{MODEL_VERSION}_{mes_scores}.csv.gz"
    final_dataset.to_csv(result_file, compression='gzip', sep=',', encoding='latin-1', index=False)
    logging.info(f"dataset {result_file} saved.")

    
