import logging
from pathlib import Path
import pandas as pd
import dask.dataframe as dd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
import sys

from ..utils import universe_creation as uc
from ..utils.config_reader import read_config

# Definition of eligible function
def eligibles(lob, client_type, month, car_filters, IND_PSIN_PCOL_ENI):
    # outras opções de inputs: par_filters, list_of_products
    # This function will filter the sources datasets to get only the eligible ones

    # Dates
    month = datetime.strptime(month,"%Y%m")
    AnoMes_Menos2 = (month + relativedelta(months=-2)).strftime("%Y%m")


    ## WRITE YOUR CODE HERE

    return ELEGIVEIS_mes_menos2

if __name__ == "__main__":


    PROJECT_PATH = Path(__file__).parents[1]
    DATA_PATH = uc.find_data_path(PROJECT_PATH) / "parquet"
    OUTPUT_PATH = PROJECT_PATH / "data"
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


    # Definition of list of months
    months = uc.generate_month(start_month, end_month)

    # Aplication of eligible function

    ELEGIVEIS=pd.DataFrame()

    for month in months:
        logging.info(f"Computing eligibles for month: {str(month)}")
        ELEGIVEIS_mes_menos2 = eligibles(
            lob, client_type, month, config.car_filters, config.IND_PSIN_PCOL_ENI) #config.par_filters, config.list_of_products
        ELEGIVEIS= pd.concat([ELEGIVEIS, ELEGIVEIS_mes_menos2], ignore_index=True)
        del ELEGIVEIS_mes_menos2


    # logging.info(ELEGIVEIS.groupby(['ANO_MES'])[put id_column here].count())
    ELEGIVEIS.to_csv(OUTPUT_PATH / f'eligibles_{start_month.strftime("%Y%m")}_{end_month.strftime("%Y%m")}.csv', index=False, encoding="latin1")