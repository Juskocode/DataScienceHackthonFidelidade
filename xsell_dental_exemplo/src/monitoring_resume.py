import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset, MonthEnd, MonthBegin
from datetime import datetime, timedelta
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols

def generate_past_months(current_ano_mes, months_back):
    year, month = int(current_ano_mes[:4]), int(current_ano_mes[4:])
    date = datetime(year, month, 1)
    dates = [(date - relativedelta(months=i)).strftime('%Y%m') for i in range(months_back + 1)]
    return dates

def load_scores(MODEL_TYPE, MODEL_LOB, MODEL_VERSION, ano_mes):
    file_name = f'SC_{MODEL_TYPE}_{MODEL_LOB}_V{MODEL_VERSION}_{ano_mes}.xlsx'
    df_scores = pd.read_excel(file_name)
    df_ordered = df_scores.sort_values(by='EM_EVENTPROBABILITY', ascending=True).reset_index(drop=True)
    return df_ordered

def load_last_n_months_scores(MODEL_TYPE, MODEL_LOB, MODEL_VERSION,current_ano_mes, n_months=5):
    past_months = generate_past_months(current_ano_mes, n_months)
    all_scores = []
    
    for month in past_months:
        try:
            df_scores = load_scores(MODEL_TYPE, MODEL_LOB, MODEL_VERSION,month)
            all_scores.append(df_scores)
        except FileNotFoundError:
            print(f"File for {month} not found.")
            continue
            
    return pd.concat(all_scores)


def merge_compras_scores(df_scores, dados_backtesting, id_column):
    
    df_model_code_compras = pd.merge(df_scores,dados_backtesting, on = [id_column] )
#     df_model_code_compras['Comp_2M'] = df_model_code_compras.apply(lambda x: x.ramosp not in x.DSC_CABAZ_ATUAL, axis=1)
#     df_model_code_compras = df_model_code_compras.loc[df_model_code_compras["Comp_2M"] == True]
    
    df_join_drc = df_model_code_compras.sort_values(by='EM_EVENTPROBABILITY', ascending=False)
    df_join_drc = df_join_drc[["ANO_MES", id_column,"IND_PSIN_PCOL_ENI", "EM_EVENTPROBABILITY","PERCENTILE"]]
    df_join_drc = df_join_drc.drop_duplicates()
    
    return df_join_drc

def calculate_metrics(df_scores, df_join_drc, id_column):
    #total scorados
    df_total_scorados = df_scores.groupby('PERCENTILE').agg(TOTAL_CUSTOMER_SCORED=(id_column, 'nunique')).reset_index()
    
    
    #capturados por percentil 
    df_captur_por_percent = df_join_drc.groupby('PERCENTILE').agg(CAPTURED=(id_column, 'count')).reset_index()
    df_captur_por_percent = df_captur_por_percent.sort_values(by='PERCENTILE')
    
    df_captur2_por_percent = df_total_scorados.merge(df_captur_por_percent, on='PERCENTILE', how='left')
    df_captur2_por_percent['CAPTURED'] = df_captur2_por_percent['CAPTURED'].fillna(0)
    
    #cumulativo capturados
    df_cumulativo_capturados = df_captur2_por_percent.copy()
    df_cumulativo_capturados['ACCUM_CAPTURED'] = df_cumulativo_capturados['CAPTURED'].cumsum()
    
    #total compras
    df_cumulativo_capturados["TOTAL_EVENTS"] = df_cumulativo_capturados["CAPTURED"].sum()    
    
    #lift
    df_cumulativo_capturados['LIFT'] = (df_cumulativo_capturados['CAPTURED'] / df_cumulativo_capturados['TOTAL_EVENTS']) * 100
    df_cumulativo_capturados['CUMULATIVE_LIFT'] = df_cumulativo_capturados['ACCUM_CAPTURED'] / (
                df_cumulativo_capturados['TOTAL_EVENTS'] / (100 / df_cumulativo_capturados['PERCENTILE']))
    
    #conv rate percentil
    df_cumulativo_capturados['CONV_RATE_PERCENTILE'] = (df_cumulativo_capturados['CAPTURED'] / df_cumulativo_capturados['TOTAL_CUSTOMER_SCORED']) * 100
    df_cumulativo_capturados['CUMULATIVE_RESPONSE'] = (df_cumulativo_capturados['ACCUM_CAPTURED'] / df_cumulativo_capturados['TOTAL_EVENTS']) * 100

    return df_cumulativo_capturados

def organizar_dataset(df_cumulativo_capturados, mes_score_ant, model_type, model_description, model_version, model_lob, review_periodicity = None):
    df_lift = df_cumulativo_capturados.copy()
    df_lift['ANO_MES'] = mes_score_ant
    df_lift['MODEL_TYPE'] = model_type
    df_lift["MODEL_DESCRIPTION"] = model_description
    df_lift['MODEL_VERSION'] = model_version
    df_lift['MODEL_LOB'] = model_lob

    if review_periodicity is None:
        columns_order = [
            'ANO_MES', 'MODEL_TYPE', 'MODEL_VERSION', 'MODEL_LOB', 
            'PERCENTILE', 'CAPTURED', 'ACCUM_CAPTURED', 'TOTAL_EVENTS', 
            'TOTAL_CUSTOMER_SCORED', 'LIFT', 'CUMULATIVE_LIFT', 'CONV_RATE_PERCENTILE', 'CUMULATIVE_RESPONSE'
        ]
    else:
        df_lift['REVIEW_PERIODICITY'] = review_periodicity
    
        columns_order = [
            'ANO_MES', 'MODEL_TYPE', 'MODEL_VERSION', 'MODEL_LOB', 'REVIEW_PERIODICITY',
            'PERCENTILE', 'CAPTURED', 'ACCUM_CAPTURED', 'TOTAL_EVENTS', 
            'TOTAL_CUSTOMER_SCORED','AVG_CUSTOMER_SCORED', 'LIFT', 'AVG_LIFTS', 'CUMULATIVE_LIFT','CUMULATIVE_LIFT_TRANSF',
            'CONV_RATE_PERCENTILE','CONV_RATE','POLINOMIAL_FUNCTION', 'LOGARITHMIC_FUNCTION', 'CONV_RATE_PERC_SMOOTHED', 'Leads_lights'
        ]
    
    return df_lift[columns_order]

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

    config = read_config(CONFIG_PATH / "scores_monitoring_config.yml")

    mes_score_ant = datetime.strftime(config.mes_scores,"%Y%m")
    id_column = config.id_column
    version = config.version
    MODEL_TYPE = config.MODEL_TYPE
    MODEL_VERSION = version[1:].split('.')[0] #getting the integer of the verion
    MODEL_LOB = config.MODEL_LOB
    
    REVIEW_PERIODICITY = config.REVIEW_PERIODICITY
    
    THRESHOLD_GREEN_6m = config.THRESHOLD_GREEN_6m
    THRESHOLD_YELLOW_6m = config.THRESHOLD_YELLOW_6m
    THRESHOLD_GREEN_12m = config.THRESHOLD_GREEN_12m
    THRESHOLD_YELLOW_12m = config.THRESHOLD_YELLOW_12m
    PERCENTIL_YELLOW = config.PERCENTIL_YELLOW
    PERCENTIL_RED = config.PERCENTIL_RED
    
    data = pd.read_csv(DATA_PATH / f'universe_{mes_score_ant}_{mes_score_ant}.csv') #alterar o nome do ficheiro para o ficheiro das compras (pode ser o Radar, nesse caso processamentos adicionais são necessários, ver xsell_cabo_verde) 
       
  
    print('Start Monitorização...')
    df_scores = load_scores(MODEL_TYPE, MODEL_LOB, MODEL_VERSION,mes_score_ant)

    df_join_drc = merge_compras_scores(df_scores, data, id_column)
    
    df_cumulativo_capturados = calculate_metrics(df_scores, df_join_drc, id_column)
    
    df_lift = organizar_dataset(df_cumulativo_capturados, mes_score_ant, MODEL_TYPE, MODEL_VERSION, MODEL_LOB)
    
    #export
    filename = f'MN_{MODEL_TYPE}_{MODEL_LOB}_V{MODEL_VERSION}_{mes_score_ant}.xlsx'
    df_lift.to_excel(DATA_PATH / filename)

    print('Start Resumo...')
    if REVIEW_PERIODICITY.lower() == "semestral":
        n_months = 5
    elif REVIEW_PERIODICITY.lower() == "anual":
        n_months = 11
    df_n_months_scores = load_last_n_months_scores(MODEL_TYPE, MODEL_LOB, MODEL_VERSION,mes_score_ant, n_months)

    df_join_drc = merge_compras_scores(df_n_months_scores, data, id_column)
    
    df_cumulativo_capturados = calculate_metrics(df_scores, df_join_drc, id_column)
    df_cumulativo_capturados = polinomial_logarithmic_functions(df_cumulativo_capturados)
    df_cumulativo_capturados2 = apply_color_coding(df_cumulativo_capturados, THRESHOLD_GREEN_12m, THRESHOLD_YELLOW_12m, PERCENTIL_YELLOW, PERCENTIL_RED)
    df_lift_periodico = organizar_dataset(df_cumulativo_capturados2, mes_score_ant, MODEL_TYPE, MODEL_VERSION, MODEL_LOB, REVIEW_PERIODICITY)

    #export
    filename = f'RS_{MODEL_TYPE}_{MODEL_LOB}_V{MODEL_VERSION}_{mes_score_ant}.xlsx'
    df_lift_periodico.to_excel(DATA_PATH / filename)

