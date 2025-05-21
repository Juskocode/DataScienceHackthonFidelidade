from pathlib import Path
import pandas as pd
import dask.dataframe as dd
from dateutil.relativedelta import relativedelta
from datetime import datetime

def find_data_path(project_path):
    current_path = project_path
    while current_path != '':  # Traverse until the root directory
        if current_path.name == 'code':
            potential_folder = current_path / 'Data'  # Check for 'Data' folder in the current directory
            if potential_folder.exists() and potential_folder.is_dir():
                return potential_folder
        current_path = current_path.parent  # Move up one directory level
    return None  # Return None if 'Data' folder is not found


# Function for definition of list of months
def generate_month(start_month, end_month):
    months = []
    current_date = start_month

    while current_date <= end_month:
        months.append(current_date.strftime("%Y%m"))
        current_date += relativedelta(months=1)

    return months


def filter_CAR_elegible(df, IND_PSIN_PCOL_ENI, conditions):
    df = df[
        (df['IND_PSIN_PCOL_ENI'].isin(IND_PSIN_PCOL_ENI)) &
        (df['FLG_CONFIDENCIALIDADE'].astype(int) == 0) &
        (df['FLG_EMPREG_FIDEL'].astype(int) == 0) &
        (df['FLG_INCIDENCIAS'].astype(int) == 0) &
        (df['FLG_CLIENTE_BLOQUEADO'].astype(int) == 0) &
        (df['FLG_FALECIDO'].astype(int) == 0) &
        (df['FLG_CONTACTO_MKT'].astype(int) == 0) &
        (df['FLG_CLIENTE_MEDIADOR'].astype(int) == 0)     
    ].copy()
    for c in conditions:
        if "'" not in c:
            col_to_convert = c.split(' ')[0]
            try:
                df[col_to_convert] = df[col_to_convert].str.replace('%', '')
            except Exception as e:
                pass
            df[col_to_convert] = df[col_to_convert].astype('float16')
    conditions = ' & '.join(conditions)
    return df.query(conditions)


# Define the PAR filtering function for selection of elegible clients (AnoMes_Menos2)
def filter_PAR_elegible(df, list_of_products, conditions):
    df = df[
        (df['FLG_APL_EM_VIGOR'].astype(int) == 1) &
        (df['COD_PRODUTO_CATALOG'] == 'INDIVIDUAL') &
        (df['DSC_PRODUTO_SO_APL'].isin(list_of_products))
    ]
    for c in conditions:
        if "'" not in c:
            col_to_convert = c.split(' ')[0]
            try:
                df[col_to_convert] = df[col_to_convert].str.replace('%', '')
            except Exception as e:
                pass
            df[col_to_convert] = df[col_to_convert].astype('float16')
    conditions = ' & '.join(conditions)
    return df.query(conditions)


# Define the PAR filtering function to identify policies that are still in effect
def filter_PAR_target0(df):
    return df[
        (df['FLG_APL_EM_VIGOR'].astype(int) == 1)
    ]


# Define the PAR filtering function to identify policies that are no longer in effect regardless of the ground for cancellation
def filter_PAR_target1G(df):
    return df[
        (df['FLG_APL_ANULADA'].astype(int) == 1)
    ]

def filter_grounds_for_cancellation(df, grounds_for_cancellation):
    return df[df['DSC_MOTIVO_ANUL_APL'].isin(grounds_for_cancellation)][['ID_APOLICE']].copy()

# Define the function to perform filtering and sampling
def filter_and_sample(df, id_column, filter_ids, sample_size):
    df = df[~df[id_column].isin(filter_ids)].copy()
    return df.sample(n=sample_size, random_state=0)
