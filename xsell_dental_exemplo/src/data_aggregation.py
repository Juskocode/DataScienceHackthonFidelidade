import logging
import os
from pathlib import Path

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from ..utils.config_reader import read_config

if __name__ == "__main__":

    PROJECT_PATH = Path(__file__).parents[1]
    DATA_PATH = PROJECT_PATH / "data"
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

    config = read_config(CONFIG_PATH / "data_aggregation_config.yml")
    historical_file = config.historical_file
    new_file = config.new_file
    id_column = config.id_column
    target_var = config.target_var_name
    universe_months = config.universe_months
    logging.info("Loading historical data...")
    if historical_file.endswith(".csv"):
        df = pd.read_csv(
            DATA_PATH / historical_file, dtype="object", encoding="latin1"
        )
    elif historical_file.endswith(".txt"):
        df = pd.read_csv(
            DATA_PATH / historical_file, sep=";", decimal=",", dtype="object", encoding="latin1"
        )
        df.rename(columns={target_var: "TARGET"}, inplace=True)
        

    df["TARGET"] = df["TARGET"].astype(int)

    logging.info("Historical data shape: %s", df.shape)
    logging.info("Loading new data...")

    df_new = pd.read_csv(
        DATA_PATH / new_file,
        sep=";",
        decimal=",",
        dtype="object",
        encoding="latin1",
        # nrows=10000,
    )

    df_new.rename(columns={target_var: "TARGET"}, inplace=True)
    df_new["TARGET"] = df_new["TARGET"].astype(int)
    
    logging.info("New data shape: %s", df_new.shape)

    new_month = datetime.strptime(df_new.ANO_MES.max(), "%Y%m")

    # Date minimum to keep in the final dataset
    min_date = datetime.strftime(new_month - relativedelta(months=universe_months), "%Y%m")

    # Filter the universe for dates superior to the min_date
    df_filtered = df.loc[df["ANO_MES"] > min_date]

    # Select IDs with Target=1 from the new month
    df_new_compras = df_new[df_new.TARGET == 1][id_column]

    # In the initial dataset, we remove all IDs Target=0 that are IDs Target=1 from the new month
    # In other words, we keep all IDs Target=1 and the IDs Target=0 that are not IDs Target=1 from the new month
    df_filtered = df_filtered.loc[~((df_filtered.TARGET == 0) & (df_filtered[id_column].isin(df_new_compras)))]

    # In the new month dataset, we remove IDs Target=0 that are IDs in the initial dataset after filtering
    # In other words, we keep all IDs Target=1 and the IDs Target=0 that are not in the initial dataset
    df_new_filtered = df_new.loc[~((df_new.TARGET == 0) & (df_new[id_column].isin(df_filtered[id_column])))]

    # To avoid having new months with much more records than the previous ones we sample
    # the data of the new month that is going to be concatenated with the previous months
    df_new_filtered = df_new_filtered.sample(
        n = int(min(df_new_filtered.shape[0], max(df.shape[0],df_new.shape[0])/universe_months)), 
        random_state = 0)

    # Aggregation of the initial dataset and the new month dataset after filtering
    abt = pd.concat([df_filtered, df_new_filtered], ignore_index=True)
    columns_not_in_df = set(df_new_filtered.columns) - set(df_filtered.columns)
    if len(columns_not_in_df) > 0:
        logging.warning(
            f"WARNING: The new file contains {len(columns_not_in_df)} new columns compared with the "
            f"historical dataset. Those columns are: {columns_not_in_df}"
        )

    logging.info("YEAR_MONTH distribution:\n%s", abt.ANO_MES.value_counts().sort_index())
    if abt.ANO_MES.nunique() != universe_months:
        logging.warning(
            f"WARNING: The aggregated table contains {abt.ANO_MES.nunique()} unique YEAR_MONTHs "
            f"but it should contain {universe_months}."
        )

    if historical_file.startswith('universe'):
        overwrite = input("Do you want to save a back-up of the historical file?[y/n]")
        if overwrite.lower() != "y" and overwrite != "":
            logging.info(f"Saving universe backup file in {DATA_PATH}/{historical_file.replace('universe','universe_bkp')}.csv...")
            df[df["ANO_MES"] <= min_date].to_csv(
                DATA_PATH / f"{historical_file.replace('universe','universe_bkp')}.csv", index=False, encoding="latin1"
            )
    
    min_date = datetime.strftime(new_month - relativedelta(months=universe_months+1), "%Y%m")
    new_file = "universe_{min_date}_{new_month}.csv"
    logging.info(f"Saving file {new_file}...")
    abt.to_csv(DATA_PATH / new_file, index=False, encoding="latin1")
    logging.info("File saved, job completed!")
