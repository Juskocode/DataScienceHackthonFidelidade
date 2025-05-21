import logging
from pathlib import Path
import pandas as pd
import dask.dataframe as dd
from dateutil.relativedelta import relativedelta
from datetime import datetime

from ..utils import universe_creation as uc
from ..utils.config_reader import read_config

def data_collection(lob, client_type, month, variables_universe):
    
    month = datetime.strptime(month,"%Y%m")
    AnoMes_Menos2 = (month + relativedelta(months=-2)).strftime("%Y%m")
    AnoMes_Menos14 = (month + relativedelta(months=-14)).strftime("%Y%m")
    
    UNIVERSO_mes_menos2 = UNIVERSO[UNIVERSO['ANO_MES'].astype(str) == AnoMes_Menos2]
    

    ## EXEMPLO DE UM CASO
    #CAR
        
    # Read dataframes
    CAR_mes_menos2 = dd.read_parquet(DATA_PATH / f'CAR_{client_type}_{AnoMes_Menos2}.parquet', engine='pyarrow')
    CAR_mes_menos2 = CAR_mes_menos2[~(CAR_mes_menos2['ID_CLIENTE_ANON'].isna())]
    CAR_mes_menos2['ID_CLIENTE_ANON'] = CAR_mes_menos2['ID_CLIENTE_ANON'].astype('int64')

    # Select features
    if 'var_car' in variables_universe.keys() and variables_universe['var_car'] != [] and variables_universe['var_car'] is not None:
        if variables_universe['var_car'] == ['all']:
            CAR_mes_menos2 = CAR_mes_menos2
        else:
            CAR_mes_menos2 = CAR_mes_menos2[variables_universe['var_car']]

    
    # Merge with Universe
    UNIVERSO_mes_menos2 = pd.merge(UNIVERSO_mes_menos2, CAR_mes_menos2.compute(), 
                              how='left', left_on='ID_CLIENTE_TM_ANON', right_on='ID_CLIENTE_ANON').drop(['ID_CLIENTE_ANON'], axis=1)
    del CAR_mes_menos2
    
    # PAR

    # Read dataframes 
    PAR_mes_menos2 = dd.read_parquet(DATA_PATH / f'PAR_{lob}_{AnoMes_Menos2}.parquet', engine='pyarrow')
    PAR_mes_menos2['ID_APOLICE'] = PAR_mes_menos2['ID_APOLICE'].astype('str')

    # Select features
    if 'var_par_sd' in variables_universe.keys() and variables_universe['var_par_sd'] != [] and variables_universe['var_par_sd'] is not None:
        if variables_universe['var_par_sd'] == ['all']:
            PAR_mes_menos2 = PAR_mes_menos2
        else:
            PAR_mes_menos2 = PAR_mes_menos2[variables_universe['var_par_sd']]


    # Merge with Universe
    UNIVERSO_mes_menos2 = pd.merge(UNIVERSO_mes_menos2, PAR_mes_menos2.compute(),
                          how='left', on='ID_APOLICE')
    del PAR_mes_menos2
    
    # PAR menos 12

    # Read dataframes 
    PAR_mes_menos14 = dd.read_parquet(DATA_PATH / f'PAR_{lob}_{AnoMes_Menos14}.parquet', engine='pyarrow')
    PAR_mes_menos14['ID_APOLICE'] = PAR_mes_menos14['ID_APOLICE'].astype('str')

    # Select features
    if 'var_par_sd_menos12' in variables_universe.keys() and variables_universe['var_par_sd_menos12'] != [] and variables_universe['var_par_sd_menos12'] is not None:
        if variables_universe['var_par_sd_menos12'] == ['all']:
            PAR_mes_menos14 = PAR_mes_menos14
        else:
            PAR_mes_menos14 = PAR_mes_menos14[variables_universe['var_par_sd_menos12']]



    # Merge with Universe
    UNIVERSO_mes_menos2 = pd.merge(UNIVERSO_mes_menos2, PAR_mes_menos14.compute(),
                          how='left', on='ID_APOLICE', suffixes=('', '_M12'))
    del PAR_mes_menos14
    
    # RAR

    # Read dataframes 
    RAR_1_mes_menos2 = dd.read_parquet(DATA_PATH / f'RAR_{lob}_1_{AnoMes_Menos2}.parquet', engine='pyarrow')
    RAR_1_mes_menos2['ID_APOLICE'] = RAR_1_mes_menos2['ID_APOLICE'].astype('str')
    RAR_2_mes_menos2 = dd.read_parquet(DATA_PATH / f'RAR_{lob}_2_{AnoMes_Menos2}.parquet', engine='pyarrow')
    RAR_2_mes_menos2['ID_APOLICE'] = RAR_2_mes_menos2['ID_APOLICE'].astype('str')
    
    # Select features
    if 'var_rar_sd' in variables_universe.keys() and variables_universe['var_rar_sd'] != [] and variables_universe['var_rar_sd'] is not None:
        if variables_universe['var_rar_sd'] == ['all']:
            RAR_1_mes_menos2 = RAR_1_mes_menos2
            RAR_2_mes_menos2 = RAR_2_mes_menos2
        else:
            RAR_1_mes_menos2 = RAR_1_mes_menos2[[c for c in variables_universe['var_rar_sd'] if c in RAR_1_mes_menos2.columns]]
            RAR_2_mes_menos2 = RAR_2_mes_menos2[[c for c in variables_universe['var_rar_sd'] if c in RAR_2_mes_menos2.columns]]

    ## FIM DO EXEMPLO DE UM CASO
    
    # Merge with Universe
    UNIVERSO_mes_menos2 = pd.merge(UNIVERSO_mes_menos2, RAR_1_mes_menos2.compute(),
                              how='left', on=['ID_APOLICE', 'COD_DELEGACAO'])
    del RAR_1_mes_menos2
    
    UNIVERSO_mes_menos2 = pd.merge(UNIVERSO_mes_menos2, RAR_2_mes_menos2.compute(),
                              how='inner', on=['ID_APOLICE', 'ID_CLIENTE_ANON', 'COD_DELEGACAO'])
    
    del RAR_2_mes_menos2
       
    
    
    # TBL_FINAL_DS

    # Read dataframes 
    DS_mes_menos2 = dd.read_parquet(DATA_PATH / f'TBL_FINAL_DS_{AnoMes_Menos2}.parquet', engine='pyarrow')
    DS_mes_menos2=DS_mes_menos2[~(DS_mes_menos2['CHAVE_ANON'].isna())]
    DS_mes_menos2['CHAVE_ANON'] = DS_mes_menos2['CHAVE_ANON'].astype('int64')

    # Select features
    if 'var_geo_ds' in variables_universe.keys() and variables_universe['var_geo_ds'] != [] and variables_universe['var_geo_ds'] is not None:
        if variables_universe['var_geo_ds'] == ['all']:
            DS_mes_menos2 = DS_mes_menos2
        else:
            DS_mes_menos2 = DS_mes_menos2[variables_universe['var_geo_ds']]

    # Merge with Universe
    UNIVERSO_mes_menos2 = pd.merge(UNIVERSO_mes_menos2, DS_mes_menos2.compute(),
                              how='left', left_on='ID_CLIENTE_TM_ANON', right_on='CHAVE_ANON').drop(['CHAVE_ANON'], axis=1)
    del DS_mes_menos2
    
    if 'columns_to_drop' in variables_universe.keys() and variables_universe['columns_to_drop'] != [] and variables_universe['columns_to_drop'] is not None:
        UNIVERSO_mes_menos2.drop(columns = variables_universe['columns_to_drop'], inplace=True)
    
    return UNIVERSO_mes_menos2
    


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
    
    variables_universe = read_config(CONFIG_PATH / "variables_universe_config.yml").variables
    
    # Definition of list of months
    months = uc.generate_month(start_month, end_month)
    
    UNIVERSO = pd.read_csv(OUTPUT_PATH / f'universe_{start_month.strftime("%Y%m")}_{end_month.strftime("%Y%m")}.csv', sep=",", encoding='latin1')
    DATA_COLLECTION=pd.DataFrame()

    for month in months:
        logging.info(f"Starting data collection for month: {str(month)}")

        # Apply data collection function
        DATA_COLLECTION_mes_menos2 = data_collection(lob, client_type, month, variables_universe)
        logging.info(DATA_COLLECTION_mes_menos2.shape)

        # Create Target1 universe
        DATA_COLLECTION = pd.concat([DATA_COLLECTION, DATA_COLLECTION_mes_menos2], ignore_index=True)


    # Export DATA_COLLECTION dataset
    DATA_COLLECTION.sort_values(by=['ANO_MES',id_column]).to_csv(OUTPUT_PATH / f'data_collection_{start_month.strftime("%Y%m")}_{end_month.strftime("%Y%m")}.csv', index=False, encoding="latin1")
