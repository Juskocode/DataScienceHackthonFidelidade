# +
# import sys
# from time import gmtime, strftime
import logging
from pathlib import Path
from pickle import dump, load

import json
import numpy as np
import pandas as pd
from pytimedinput import timedInput
from dateutil.relativedelta import relativedelta
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#from sklearn.compose import ColumnTransformer
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import OneHotEncoder

from ..utils import universe_creation as uc
from ..utils.config_reader import read_config
from ..utils import data_processing as dp
from ..utils import feature_mapping as feat_map

# Suppress specific UserWarning
warnings.filterwarnings("ignore", message="The number of unique categories for variable")

if __name__ == "__main__":

    PROJECT_PATH = Path(__file__).parents[1]
    LABEL_PATH = uc.find_data_path(PROJECT_PATH) / "metadata"
    DATA_PATH = PROJECT_PATH / "data"
    REPORT_PATH = PROJECT_PATH / "report"
    AUX_REPORT_PATH = PROJECT_PATH / "report" / "Auxiliary Reports"
    MODEL_PATH = PROJECT_PATH / "models"
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

    config = read_config(CONFIG_PATH / "create_universe_config.yml")
    lob = config.lob
    client_type = config.client_type
    start_month = config.start_month
    end_month = config.end_month

    config2 = read_config(CONFIG_PATH / "prepare_data_config.yml")

    id_column = config2.id_column
    version = config2.version
    test_size = config2.test_size
    columns_to_drop = config2.columns_to_drop

    special_columns = [id_column, "ANO_MES"]
    
    with open(AUX_REPORT_PATH / "feature_mapping.json") as file:
        feature_mapping = json.load(file)
        feature_mapping[version] = {}

    ## ADD OR REMOVE THE TABLES THAT ARE NEEDED
    # Import table labels, insert tables that are used
    car_labels = pd.read_excel(LABEL_PATH / f'Variáveis CAR_{client_type}.xlsx', index_col = 'Name')
    #car_labels.drop('ID_CLIENTE_ANON',inplace=True) # we want to use the label from RAR_SD
    car_labels['Source'] = 'CAR'
    
    par_sd_labels = pd.read_excel(LABEL_PATH / f'Variáveis PAR_{lob}.xlsx', index_col = 'Name')
    par_sd_labels['Source'] = f'PAR_{lob}'
    
    rar_sd_labels = pd.read_excel(LABEL_PATH / f'Variáveis RAR_{lob}.xlsx', index_col = 'Name')
    rar_sd_labels['Source'] = f'RAR_{lob}'
    
    # Put them in importance order to keep the one of interest
    labels = pd.concat([car_labels,par_sd_labels,rar_sd_labels],axis=0)
    labels = labels[~labels.index.duplicated(keep='first')]
    ## END OF DEFINING THE TABLES THAT ARE NEEDED

    # Import data
    logging.info("Loading input data...")

    if config2.input_file is None or config2.input_file == "":
        data = pd.read_csv(DATA_PATH / f'data_collection_{start_month.strftime("%Y%m")}_{end_month.strftime("%Y%m")}.csv', sep=",",dtype='object', encoding='latin1')
    else:
        data = pd.read_csv(DATA_PATH / config2.input_file, sep=";", decimal=",", dtype='object', encoding='latin1')
    
    #split train-test
    if test_size >= 1:
        test_month = int((end_month - relativedelta(months=test_size) - relativedelta(months=2)).strftime("%Y%m"))
        train=data.loc[(data['ANO_MES'].astype(int)<=test_month)]
        test=data.loc[(data['ANO_MES'].astype(int)>test_month)]
    else:
        train, test = train_test_split(data, random_state=10, test_size=test_size)
        
    logging.info("Saving test data...\n")    
    test.to_csv(
        DATA_PATH / f"original_testing_data_{version.replace('.','-')}.csv",
        encoding="latin1",
        index=False,
    )        
    
    del data
    del test

    train = train[["COD_TIPO_ID_CLIENTE", "IND_PESSOA_SING_COLECTIVA_ENI", "COD_SEGMENTO_CLIENTE_MKT", "FLG_CONTACTO_MKT",
                   "FLG_EMPREG_FIDEL","FLG_CLIENTE_MEDIADOR", "",
                   "TARGET"] + special_columns] ### ADD HERE ALL OF THE ORIGINAL VARIABLES OF THE MODEL FINAL VARIABLES OF THE MODEL

    logging.info("Processing train data...\n")

    ## ADD HERE ALL TRANSFORMATIONS YOU NEED TO TRANSFORM YOUR TRAIN DATA AND SAVE THOSE TRANSFORMATIONS FOR FURTHER USE



    ## END OF TRAINING DATA TRANSFORMATIONS

    #Saving Feature Mapping
    with open(AUX_REPORT_PATH / "feature_mapping.json", "w") as file:
        json.dump(feature_mapping, file, ensure_ascii=False)

    logging.info("Saving training data...\n")
    pd.concat([X_special, X_train, y_train], axis=1).to_csv(
        DATA_PATH / f"training_data_{version.replace('.','-')}.csv",
        encoding="latin1",
        index=False,
    )
    logging.info("Job completed!")
