# +
# import sys
# from time import gmtime, strftime
import logging
import warnings
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import universe_creation as uc
from ..utils.config_reader import read_config
from ..utils import data_processing as dp
from ..utils import feature_mapping as feat_map

# Suppress specific UserWarning
warnings.filterwarnings("ignore", message="The number of unique categories for variable")

if __name__ == "__main__":

    PROJECT_PATH = Path(__file__).parents[1]
    #LABEL_PATH = uc.find_data_path(PROJECT_PATH) / "metadata"
    DATA_PATH = PROJECT_PATH / "data"
    LABEL_PATH = DATA_PATH / "metadata"
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
    #car_labels = pd.read_excel(LABEL_PATH / f'Variáveis CAR_{client_type}.xlsx', index_col = 'Name')
    #car_labels.drop('ID_CLIENTE_ANON',inplace=True) # we want to use the label from RAR_SD
    #car_labels['Source'] = 'CAR'
    
    #par_sd_labels = pd.read_excel(LABEL_PATH / f'Variáveis PAR_{lob}.xlsx', index_col = 'Name')
    #par_sd_labels['Source'] = f'PAR_{lob}'
    
    #rar_sd_labels = pd.read_excel(LABEL_PATH / f'Variáveis RAR_{lob}.xlsx', index_col = 'Name')
    #rar_sd_labels['Source'] = f'RAR_{lob}'
    
    # Put them in importance order to keep the one of interest
    #labels = pd.concat([car_labels,par_sd_labels,rar_sd_labels],axis=0)
    #labels = labels[~labels.index.duplicated(keep='first')]

    dental_labels = pd.read_excel(LABEL_PATH / 'VariaveisBIC_PRINCIPAL.xlsx', index_col='Name')
    dental_labels['Source'] = 'DENTAL'

    labels = pd.concat([dental_labels], axis=0)
    labels = labels[~labels.index.duplicated(keep='first')]
    ## END OF DEFINING THE TABLES THAT ARE NEEDED

    # Import data
    logging.info("Loading input data...")

    #if config2.input_file is None or config2.input_file == "":
    data = pd.read_csv(DATA_PATH / f'data_collection_{start_month.strftime("%Y%m")}_{end_month.strftime("%Y%m")}.csv', sep=",",dtype='object', encoding='latin1')
    #else:
        #data = pd.read_csv(DATA_PATH / config2.input_file, sep=";", decimal=",", dtype='object', encoding='latin1')

    #print(data['TARGET'])
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

    logging.info("Processing train data...\n")

    ## ADD HERE ALL TRANSFORMATIONS YOU NEED TO TRANSFORM YOUR TRAIN DATA AND SAVE THOSE TRANSFORMATIONS FOR FURTHER USE
    # Import table labels, insert tables that are used
    # At the beginning, before any transformations, check if the columns exist
    logging.info(f"Available columns in the train dataset: {train.columns.tolist()}")

    # Make sure that special_columns are defined based on what's available in the data
    # Check the scores_monitoring_config.yml - id_column is empty there
    # We need to ensure id_column has a valid value
    if not id_column:  # If id_column is empty string or None
        # Look for candidate id columns in the data
        id_candidates = [col for col in train.columns if 'ID' in col or 'id' in col]
        if id_candidates:
            id_column = id_candidates[0]  # Use the first ID column
            logging.info(f"Using {id_column} as the ID column")
        else:
            # If no ID column found, create a simple index
            train['row_id'] = train.index
            id_column = 'row_id'
            logging.info(f"No ID column found, creating {id_column}")

    # Also check if ANO_MES exists
    if 'ANO_MES' not in train.columns:
        # Create a default ANO_MES column if missing
        current_date = datetime.now()
        train['ANO_MES'] = int(current_date.strftime("%Y%m"))
        logging.info("ANO_MES column not found, created with current date")

    # Update special_columns with the correct column names
    special_columns = [id_column, "ANO_MES"]
    logging.info(f"Using special columns: {special_columns}")

    # Verify these columns exist before transformation
    for col in special_columns:
        if col not in train.columns:
            raise ValueError(f"Required column {col} not found in the data")

    # Now proceed with your transformers
    logging.info("Fitting and saving first processor...\n")
    processor1 = Pipeline([
        ("parse_data", dp.ParseDataTransformer(id_column, labels['Type'])),
        ("feature_transformer", dp.FeatureTransformerCar()),
    ])

    logging.info("Fitting first processor...\n")
    train1 = processor1.fit_transform(train)

    # Check what columns are available after transformation
    logging.info(f"Available columns after transformation: {train1.columns.tolist()}")

    # Handle missing special columns if the transformation removed them
    for col in special_columns:
        if col not in train1.columns:
            logging.warning(f"Column {col} was removed during transformation, attempting to add it back")
            # Copy back the column from the original data
            train1[col] = train[col].values

    for p in processor1:
        if hasattr(p, "feature_mapping"):
            feature_mapping[version].update(p.feature_mapping)

    logging.info("Saving first processor...\n")
    dump(
        processor1,
        open(MODEL_PATH / "preprocess" / f"processor1_{version.replace('.', '-')}.pkl", "wb"),
    )
    del processor1
    del train

    # Verify special columns exist before extracting
    for col in special_columns:
        if col not in train1.columns:
            raise ValueError(f"Required column {col} not found after processing")

    X_special = train1[special_columns].copy()
    X_train = train1.drop(columns=special_columns + ["TARGET"])
    y_train = train1["TARGET"]

    X_special = train1[special_columns].copy()
    X_train = train1.drop(columns=special_columns + ["TARGET"])
    y_train = train1["TARGET"]

    logging.info("Fitting and saving second processor...\n")
    processor2 = Pipeline([
        ("uniformize_names", dp.UniformizeNamesTransformer(renaming_dict={'IND_PSIN_PCOL_ENI': 'IND_PSIN_PCOL_ENI'})),
        #("AntCliente", dp.FillMissingAntCliente(version, AUX_REPORT_PATH)),
        ("drop_columns", dp.ColumnSelectorTransformer(columns_to_drop=[], missing_threshold=0.97)),
        # drop irrelevant columns and high missing values
    ])

    logging.info("Fitting second processor...\n")
    X_train = processor2.fit_transform(X_train)

    for p in processor2:
        if hasattr(p, "feature_mapping"):
            feature_mapping[version].update(p.feature_mapping)

    logging.info("Saving second processor...\n")
    dump(
        processor2,
        open(MODEL_PATH / "preprocess" / f"processor2_{version.replace('.', '-')}.pkl", "wb"),
    )
    del processor2
    del train1

    median_fill_columns = [c for c in X_train.columns if ("_DICO" in c) or (c in [
        "QTD_IDADE",
        "VAL_DENS_POP_CONC_INE",
        "Margem_gbm",
        "VAL_SCORE_FNOL",
        "CLV",
        "VAL_SCORE_CLV",
        "VAL_MARG_TECNICA_3A",
    ])]

    ordinal_cols = ["FLG_RURAL_URBANO"] if "FLG_RURAL_URBANO" in X_train.columns else []
    ordinal_categories = {
        "FLG_RURAL_URBANO": [
            [
                "ÁREA PREDOMINANTEMENTE URBANA",
                "ÁREA MEDIAMENTE URBANA",
                "ÁREA PREDOMINANTEMENTE RURAL",
                "INDETERMINADO",
            ]
        ]
    }

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.drop(ordinal_cols).tolist()
    numeric_columns = X_train.select_dtypes(include='number').columns
    numeric_columns_clean = [col for col in numeric_columns if col not in median_fill_columns]

    logging.info("Fitting and saving third processor...\n")
    processor3 = Pipeline([
        (
            "fill_transform",
            ColumnTransformer(
                transformers=[
                    (
                        "median_fill",
                        dp.InputMedian(
                            version,
                            AUX_REPORT_PATH,
                            categorical_strategy='most_frequent'
                        ),
                        median_fill_columns,
                    ),
                    (
                        "ordinalencoder",
                        dp.OrdinalManualEncoder(
                            column_categories=ordinal_categories,
                            auto_detect=True,
                            handle_unseen="encode"
                        ),
                        ordinal_cols,
                    ),
                    (
                        "numfillna",
                        SimpleImputer(strategy="constant", fill_value=-99999),
                        numeric_columns_clean,
                    ),
                    (
                        "catfillna",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                        categorical_cols,
                    ),
                ],
                remainder="passthrough",
                verbose_feature_names_out=False,
            ).set_output(transform="pandas"),
        ),
        ('target_encoder', dp.TargetManualEncoder(AUX_REPORT_PATH, version)),
    ])
    logging.info("Fitting and saving feature aggregation processor...\n")
    feature_aggregator = ComprehensiveFeatureTransformer(
        target_column='TARGET',
        min_correlation=0.02,
        coherence_threshold=0.3
    )

    logging.info("Fitting feature aggregator...\n")
    X_train = feature_aggregator.fit_transform(X_train, y_train)

    if hasattr(feature_aggregator, "feature_mapping"):
        feature_mapping[version].update(feature_aggregator.feature_mapping)

    logging.info("Saving feature aggregator...\n")
    dump(
        feature_aggregator,
        open(MODEL_PATH / "preprocess" / f"feature_aggregator_{version.replace('.', '-')}.pkl", "wb"),
    )

    # Categorical features processing
    logging.info("Fitting and saving categorical feature processor...\n")
    cat_processor = CategoricalFeatureTransformer(
        target_column='TARGET',
        ordinal_columns=ordinal_cols,
        ordinal_mappings=ordinal_categories,
        high_cardinality_threshold=0.1
    )

    logging.info("Fitting categorical processor...\n")
    X_train = cat_processor.fit_transform(X_train, y_train)

    if hasattr(cat_processor, "feature_mapping"):
        feature_mapping[version].update(cat_processor.feature_mapping)

    logging.info("Saving categorical processor...\n")
    dump(
        cat_processor,
        open(MODEL_PATH / "preprocess" / f"cat_processor_{version.replace('.', '-')}.pkl", "wb"),
    )

    # Imputation and scaling
    logging.info("Fitting and saving imputation and scaling processor...\n")
    impute_scale_processor = ImputationAndScalingTransformer(
        numerical_imputation_strategy='median',
        high_missing_threshold=0.3,
        scaling_method=None  # Set to 'standardization' or 'minmax' if scaling is needed
    )

    logging.info("Fitting imputation and scaling processor...\n")
    X_train = impute_scale_processor.fit_transform(X_train)

    if hasattr(impute_scale_processor, "feature_mapping"):
        feature_mapping[version].update(impute_scale_processor.feature_mapping)

    logging.info("Saving imputation and scaling processor...\n")
    dump(
        impute_scale_processor,
        open(MODEL_PATH / "preprocess" / f"impute_scale_processor_{version.replace('.', '-')}.pkl", "wb"),
    )

    logging.info("Fitting third processor...\n")
    X_train = processor3.fit_transform(X_train, y_train)

    for p in processor3:
        if hasattr(p, "feature_mapping"):
            feature_mapping[version].update(p.feature_mapping)

    logging.info("Saving third processor...\n")
    dump(
        processor3,
        open(MODEL_PATH / "preprocess" / f"processor3_{version.replace('.', '-')}.pkl", "wb"),
    )
    del processor3
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

