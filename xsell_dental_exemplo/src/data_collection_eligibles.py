import logging
import json
from pathlib import Path
import pandas as pd
import dask.dataframe as dd
from dateutil.relativedelta import relativedelta
from datetime import datetime

from ..utils import universe_creation as uc
from ..utils.config_reader import read_config
from ..utils.feature_mapping import group_transformations

def original_features(lob, client_type, version):

    with open(AUX_REPORT_PATH / "selected_features.json") as file:
        selected_features = json.load(file)
        selected_features = selected_features[version] 
    with open(AUX_REPORT_PATH / "feature_mapping.json") as file:
        feature_mapping = json.load(file)   
        feature_mapping = feature_mapping[version]    
    cols_to_use = []
    for v in selected_features:
        if v not in feature_mapping.keys():
            cols_to_use.append(v)

        else:
            cols_to_use += group_transformations(feature_mapping[v])[0][0]
#    for v in feature_mapping.keys():
#        cols_to_use += group_transformations(feature_mapping[v])[0][0]

    # Import table labels, insert tables that are used
    car_labels = pd.read_excel(LABEL_PATH / f'Variáveis CAR_{client_type}.xlsx', index_col = 'Name')
    #car_labels.drop('ID_CLIENTE_ANON',inplace=True) # we want to use the label from RAR_SD
    car_labels['Source'] = 'CAR'
    
    par_sd_labels = pd.read_excel(LABEL_PATH / f'Variáveis PAR_{lob}.xlsx', index_col = 'Name')
    par_sd_labels['Source'] = f'PAR_{lob}'
    
    rar_sd_labels = pd.read_excel(LABEL_PATH / f'Variáveis RAR_{lob}.xlsx', index_col = 'Name')
    rar_sd_labels['Source'] = f'RAR_{lob}'

    tbl_sd_labels = pd.read_excel(LABEL_PATH / 'Variáveis CHURN_SAUDE_TBL_FINAL_DS.xlsx', index_col = 'Name')
    tbl_sd_labels['Source'] = 'CHURN_SAUDE_TBL'
    
    # Put them in importance order to keep the one of interest
    labels = pd.concat([car_labels,par_sd_labels,rar_sd_labels,tbl_sd_labels],axis=0)
    labels = labels[~labels.index.duplicated(keep='first')]

    for c in cols_to_use:
        if c not in labels.index:
            col_prefix = c.rsplit('_',1)[0]
            if col_prefix in labels.index:
                logging.warning(
                        f"WARNING: Column {c} is not labeled in the labels dataset. So you should get column {col_prefix} for {c}."
                    )
                cols_to_use.append(col_prefix)
            else:
                logging.warning(f"{c} is not labeled so you will get an error.")

    return labels.loc[list(set(cols_to_use)),:]

        

def data_collection_eligibles(lob, client_type, month, version):
    
    month = datetime.strptime(month,"%Y%m")
    AnoMes_Menos2 = (month + relativedelta(months=-2)).strftime("%Y%m")
    AnoMes_Menos14 = (month + relativedelta(months=-14)).strftime("%Y%m")
    

    ## EXEMPLO DE UM CASO
    labels = original_features(lob, client_type, version)
    #CAR
        
    # Read dataframes
    CAR_mes_menos2 = dd.read_parquet(DATA_PATH / f'CAR_{client_type}_{AnoMes_Menos2}.parquet', engine='pyarrow')
    CAR_mes_menos2 = CAR_mes_menos2[~(CAR_mes_menos2['ID_CLIENTE_ANON'].isna())]
    CAR_mes_menos2['ID_CLIENTE_ANON'] = CAR_mes_menos2['ID_CLIENTE_ANON'].astype('int64')

    # Select features
    CAR_mes_menos2 = CAR_mes_menos2[labels[labels['Source']=='CAR'].index.tolist()+['ID_CLIENTE_ANON']]

    
    # Merge with Universe
    ELEGIVEIS = pd.merge(ELEGIVEIS, CAR_mes_menos2.compute(), 
                              how='left', left_on='ID_CLIENTE_TM_ANON', right_on='ID_CLIENTE_ANON').drop(['ID_CLIENTE_ANON'], axis=1)
    del CAR_mes_menos2
    
    # PAR

    # Read dataframes 
    PAR_mes_menos2 = dd.read_parquet(DATA_PATH / f'PAR_{lob}_{AnoMes_Menos2}.parquet', engine='pyarrow')
    PAR_mes_menos2['ID_APOLICE'] = PAR_mes_menos2['ID_APOLICE'].astype('str')

    # Select features
    PAR_mes_menos2 = PAR_mes_menos2[labels[labels['Source']==f'PAR_{lob}'].index.tolist()+['ID_APOLICE']]


    # Merge with Universe
    ELEGIVEIS = pd.merge(ELEGIVEIS, PAR_mes_menos2.compute(),
                          how='left', on='ID_APOLICE')
    del PAR_mes_menos2
    
    # PAR menos 12

    # Read dataframes 
    PAR_mes_menos14 = dd.read_parquet(DATA_PATH / f'PAR_{lob}_{AnoMes_Menos14}.parquet', engine='pyarrow')
    PAR_mes_menos14['ID_APOLICE'] = PAR_mes_menos14['ID_APOLICE'].astype('str')

    # Select features
    if 'var_par_sd_menos12' in variables_universe.keys() and variables_universe['var_par_sd_menos12'] != [] and variables_universe['var_par_sd_menos12'] is not None:
        PAR_mes_menos14 = PAR_mes_menos14[+['ID_APOLICE']]


    # Merge with Universe
    ELEGIVEIS = pd.merge(ELEGIVEIS, PAR_mes_menos14.compute(),
                          how='left', on='ID_APOLICE', suffixes=('', '_M12'))
    del PAR_mes_menos14
    
    # RAR

    # Read dataframes 
    RAR_1_mes_menos2 = dd.read_parquet(DATA_PATH / f'RAR_{lob}_1_{AnoMes_Menos2}.parquet', engine='pyarrow')
    RAR_1_mes_menos2['ID_APOLICE'] = RAR_1_mes_menos2['ID_APOLICE'].astype('str')
    RAR_2_mes_menos2 = dd.read_parquet(DATA_PATH / f'RAR_{lob}_2_{AnoMes_Menos2}.parquet', engine='pyarrow')
    RAR_2_mes_menos2['ID_APOLICE'] = RAR_2_mes_menos2['ID_APOLICE'].astype('str')
    
    # Select features
    RAR_1_mes_menos2 = RAR_1_mes_menos2[[c for c in labels[labels['Source']==f'RAR_{lob}'].index.tolist() if c in RAR_1_mes_menos2.columns]+['ID_APOLICE', 'ID_CLIENTE_ANON', 'COD_DELEGACAO']]
    RAR_2_mes_menos2 = RAR_2_mes_menos2[[c for c in labels[labels['Source']==f'RAR_{lob}'].index.tolist() if c in RAR_2_mes_menos2.columns]+['ID_APOLICE', 'ID_CLIENTE_ANON', 'COD_DELEGACAO']]

    
    # Merge with Universe
    ELEGIVEIS = pd.merge(ELEGIVEIS, RAR_1_mes_menos2.compute(),
                              how='left', on=['ID_APOLICE', 'COD_DELEGACAO'])
    del RAR_1_mes_menos2
    
    ELEGIVEIS = pd.merge(ELEGIVEIS, RAR_2_mes_menos2.compute(),
                              how='inner', on=['ID_APOLICE', 'ID_CLIENTE_ANON', 'COD_DELEGACAO'])
    
    del RAR_2_mes_menos2
       
    
    # TBL_FINAL_DS

    # Read dataframes 
    DS_mes_menos2 = dd.read_parquet(DATA_PATH / f'TBL_FINAL_DS_{AnoMes_Menos2}.parquet', engine='pyarrow')
    DS_mes_menos2=DS_mes_menos2[~(DS_mes_menos2['CHAVE_ANON'].isna())]
    DS_mes_menos2['CHAVE_ANON'] = DS_mes_menos2['CHAVE_ANON'].astype('int64')

    # Select features
    DS_mes_menos2 = DS_mes_menos2[labels[labels['Source']=='CHURN_SAUDE_TBL'].index.tolist()+['CHAVE_ANON']]

    # Merge with Universe
    ELEGIVEIS = pd.merge(ELEGIVEIS, DS_mes_menos2.compute(),
                              how='left', left_on='ID_CLIENTE_TM_ANON', right_on='CHAVE_ANON').drop(['CHAVE_ANON'], axis=1)
    del DS_mes_menos2

    ## FIM DO EXEMPLO DE UM CASO
    
    
    return ELEGIVEIS
    


if __name__ == "__main__":


    PROJECT_PATH = Path(__file__).parents[1]
    LABEL_PATH = uc.find_data_path(PROJECT_PATH) / "metadata"
    DATA_PATH = uc.find_data_path(PROJECT_PATH) / "parquet"
    OUTPUT_PATH = PROJECT_PATH / "data"
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


    config = read_config(CONFIG_PATH / "create_universe_config.yml")
    lob = config.lob
    client_type = config.client_type
    start_month = config.start_month
    end_month = config.end_month

    variables_universe = read_config(CONFIG_PATH / "variables_universe_config.yml").variables
    
    config2 = read_config(CONFIG_PATH / "scores_monitoring_config.yml")
    version = config2.version
    id_column = config2.id_column
    MODEL_TYPE = config2.MODEL_TYPE
    MODEL_VERSION = version[1:].split('.')[0] #getting the integer of the verion
    MODEL_LOB = config2.MODEL_LOB
    
    # Definition of list of months
    months = uc.generate_month(start_month, end_month)
    
    ELEGIVEIS = pd.read_csv(OUTPUT_PATH / f'eligibles_{start_month.strftime("%Y%m")}_{end_month.strftime("%Y%m")}.csv', sep=",", encoding='latin1')
    DATA_COLLECTION=pd.DataFrame()

    for month in months:
        logging.info(f"Starting data collection for month: {str(month)}")

        # Apply data collection function
        DATA_COLLECTION_mes_menos2 = data_collection_eligibles(ELEGIVEIS, lob, client_type, month, version)
        logging.info(DATA_COLLECTION_mes_menos2.shape)

        # Create Target1 universe
        DATA_COLLECTION = pd.concat([DATA_COLLECTION, DATA_COLLECTION_mes_menos2], ignore_index=True)


    # Export DATA_COLLECTION_ELIGIBLES dataset
    output_file = f"ELEG_{MODEL_TYPE}_{MODEL_LOB}_V{MODEL_VERSION}_{end_month.strftime('%Y%m')}.csv.gz"
    DATA_COLLECTION.sort_values(by=['ANO_MES',id_column]).to_csv(OUTPUT_PATH / output_file, index=False, encoding="latin1", compression='gzip')
