import json
import logging
import os
import warnings
from collections import defaultdict, Counter
from time import gmtime, strftime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import expon, poisson
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

# Configure logging for the external script
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
formatter = logging.Formatter("%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class ParseDataTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for transforming data types from a DataFrame based on the labels from SAS tables.

    This class categorizes DataFrame columns by their types and applies appropriate transformations,
    such as replacing special characters, converting data types, and dropping some columns.

    Methods:
    --------
    fit(X: pd.DataFrame):
        Fits the transformer to the DataFrame. Required for pipeline compatibility but does not perform any action.

    transform(X: pd.DataFrame) -> pd.DataFrame:
        Transforms the DataFrame based on the labels.

    fit_transform(X: pd.DataFrame) -> pd.DataFrame:
        Fits the transformer to the DataFrame and then transforms it.
    """

    def __init__(self, id_column, labels_df):
        self.id_column = id_column
        self.labels_df = pd.DataFrame(labels_df)

    def _categorize_columns(self, X):
        """Categorize columns based on their prefixes."""
        # Initialize the column categories
        self.cat_cols = []
        self.num_cols = []
        self.dt_cols = []
        
        self.tho_sep_cols = []
        thousand_sep_pattern = r'\d{1,3}(?:\.\d{3})+'
        self.decsep_cols = []
        decimal_sep_pattern = r'\d+,\d+'
        self.spec_chr_cols = []
        special_char_pattern = r'[()%*"\'%]'
        
        self.id_cols = []
        self.nif_cols = []

        for c in X.columns:
            if c.startswith("ID_") and c != self.id_column:
                self.id_cols.append(c)
            elif c.startswith("NIF"):
                self.nif_cols.append(c)
                
            elif c in self.labels_df.index:
                if self.labels_df.loc[c,'Type'] == 'Numeric':
                    sample_values = X[c].astype(str).drop_duplicates()
                    # Check for thousand separator pattern (e.g., 1.234 or 12.345)
                    if any(sample_values.str.contains(thousand_sep_pattern, regex=True)) and any(sample_values.str.contains(decimal_sep_pattern, regex=True)):
                        self.tho_sep_cols.append(c)

                    # Check for decimal separator pattern (e.g., 1234,56)
                    if any(sample_values.str.contains(decimal_sep_pattern, regex=True)):
                        self.decsep_cols.append(c)

                    # Check for special characters pattern (e.g., %, $, etc.)
                    if any(sample_values.str.contains(special_char_pattern, regex=True)):
                        self.spec_chr_cols.append(c)
                        
                    self.num_cols.append(c)


                elif self.labels_df.loc[c,'Type'] == 'Date':
                    self.dt_cols.append(c)
                elif self.labels_df.loc[c,'Type'] == 'Character':
                    self.cat_cols.append(c)
                else:
                    logging.warning(
                        f"WARNING: Column {c} has unknown label {self.labels_df.loc[c,'Type']} in the labels dataset." 
                        "So it will be considered as object."
                    )

            elif c == 'TARGET':
                self.num_cols.append(c)
            else:
                col_prefix = c.rsplit('_',1)[0]
                if col_prefix in X.columns:
                    if col_prefix in self.tho_sep_cols:
                        self.tho_sep_cols.append(c)
                    if col_prefix in self.decsep_cols:
                        self.decsep_cols.append(c)
                    if col_prefix in self.num_cols:
                        self.num_cols.append(c)
                    if col_prefix in self.dt_cols:
                        self.dt_cols.append(c)
                    if col_prefix in self.cat_cols:
                        self.cat_cols.append(c)
                    logging.warning(
                        f"WARNING: Column {c} is not labeled in the labels dataset. So it will be transformed like column {col_prefix}."
                    )
                else:    
                    logging.warning(
                        f"WARNING: Column {c} is not labeled in the labels dataset. So it will be considered as object."
                    )
              

    def _apply_transformations(self, X):
        """Apply the necessary transformations to the DataFrame."""
        X_transformed = X.copy()
        #with pd.option_context('future.no_silent_downcasting', True): 
        X_transformed.loc[:, self.tho_sep_cols] = X_transformed[self.tho_sep_cols].replace(to_replace=r"\.", value="", regex=True)
        X_transformed.loc[:, self.decsep_cols] = X_transformed[self.decsep_cols].replace(to_replace=r",", value=".", regex=True)
        X_transformed.loc[:, self.spec_chr_cols] = X_transformed[self.spec_chr_cols].replace(to_replace=r"[()*\"'%]", value="", regex=True)
        X_transformed.loc[:, self.cat_cols] = X_transformed[self.cat_cols].astype("category")
        X_transformed.loc[:, self.num_cols] = X_transformed[self.num_cols].replace(
            to_replace=r"^$", value=np.nan, regex=True
        )
        # here it only works without .loc
        X_transformed[self.dt_cols] = X_transformed[self.dt_cols].apply(pd.to_datetime, format="%Y-%m-%d", errors='coerce')

        for c in self.num_cols:
            X_transformed[c] = pd.to_numeric(X_transformed[c], errors="raise").astype("float64")
        X_transformed.drop(columns=self.nif_cols+self.id_cols, inplace=True, errors="ignore")
        #X_transformed[id_column] = X_transformed[id_column].astype('object')
        return X_transformed

    def fit(self, X, y=None):
        """Fit method, required by sklearn pipeline but not used in this transformer."""
        return self

    def transform(self, X):
        """Transform the DataFrame based on the fitted categories."""
        self._categorize_columns(X)
        X_transformed = self._apply_transformations(X)
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        self.fit(X,y)
        return self.transform(X)

class UniformizeNamesTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer to uniformize column names in a dataset.

    Parameters:
    -----------
    renaming_dict : dict, optional (default={})
        A dictionary containing original column names as keys and their corresponding
        renamed column names as values.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data.

    transform(X):
        Transform the input data by renaming columns.

    fit_transform(X, y=None):
        Fit the transformer to the data and transform it.

    get_feature_names_out(input_features=None):
        Get the names of the output features after transformation.
    """

    def __init__(self, renaming_dict={}):
        self.renamed_columns = []
        self.dropped_columns = []
        self.renaming_dict = renaming_dict
        
    def create_renaming_dict(self, X):
        self.dropped_columns = []
        for col in X.columns:
            if col not in self.renaming_dict.keys():
                self.renaming_dict[col] = col.replace(" - ", "_").replace(" ", "_").replace("_-_", "_")
                if self.renaming_dict[col].endswith("_nan") or self.renaming_dict[col].endswith("_0"):
                    self.dropped_columns.append(col)
                # Uniformize quantity columns
                elif self.renaming_dict[col].startswith("QT_"):
                    self.renaming_dict[col] = self.renaming_dict[col].replace("QT_", "QTD_")
                elif self.renaming_dict[col].startswith("QTY_"):
                    self.renaming_dict[col] = self.renaming_dict[col].replace("QTY_", "QTD_")
                # Uniformize percentage columns
                elif self.renaming_dict[col].startswith("PCT_"):
                    self.renaming_dict[col] = self.renaming_dict[col].replace("PCT_", "PERC_")
                # Uniformize FLAG columns
                elif self.renaming_dict[col].startswith("FLAG_"):
                    self.renaming_dict[col] = self.renaming_dict[col].replace("FLAG_", "FLG_")
                elif self.renaming_dict[col].startswith("COD_"):
                    self.renaming_dict[col] = self.renaming_dict[col].replace("COD_", "FLG_")
                #elif self.renaming_dict[col].startswith("DSC_"):
                #    self.renaming_dict[col] = self.renaming_dict[col].replace("DSC_", "FLG_")
                #elif self.renaming_dict[col].startswith("OD_STANDARDIZED_"):
                 #   self.renaming_dict[col] = self.renaming_dict[col].replace("OD_STANDARDIZED_", "FLG_")
                elif self.renaming_dict[col].startswith("IND_"):
                    self.renaming_dict[col] = self.renaming_dict[col].replace("IND_", "FLG_")

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        self.create_renaming_dict(X)
        X_transformed = X.drop(columns=self.dropped_columns)
        X_transformed = X_transformed.rename(columns=self.renaming_dict, errors = 'ignore')
        self.renamed_columns = X_transformed.columns.tolist()
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.renamed_columns

# Custom transformer to split the data to be aggregated
class DataSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, id_column, special_columns, labels_df, source_table):
        self.id_column = id_column
        self.special_columns = special_columns
        self.labels_df = labels_df
        self.source_table = source_table

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Split the data
        df_to_agg = X[[c for c in X.columns if (c in self.labels_df[self.labels_df['Source']==self.source_table].index and c not in self.special_columns)]+self.special_columns]

        df_train = X[[c for c in X.columns if (c not in self.labels_df[self.labels_df['Source']==self.source_table].index and c not in self.special_columns)]+self.special_columns].drop_duplicates()

        n_duplicates = (df_train[self.id_column].value_counts()>1).sum()
        if n_duplicates > 0:
            logging.warning(f"WARNING: There are {n_duplicates} {self.id_column} duplicated.")
        return df_to_agg.set_index(self.special_columns), df_train.set_index(self.special_columns)
        
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        self.fit(X,y)
        return self.transform(X)

class DataMerger(BaseEstimator, TransformerMixin):
    def __init__(self, id_column):
        self.id_column = id_column

    def fit(self, X, y=None):
        return self

    def transform(self, X_parts):
        part1, part2 = X_parts
        # Merge on the ID column
        merged_data = pd.merge(part1, part2, left_index=True, right_index=True, how='inner') #on=self.id_column
        return merged_data
    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        self.fit(X,y)
        return self.transform(X)

class FeatureTransformerCar(BaseEstimator, TransformerMixin):
    """
    A transformer to perform feature engineering on the CAR dataset.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data.

    transform(X):
        Transform the input data by applying feature engineering.

    fit_transform(X, y=None):
        Fit the transformer to the data and transform it.

    get_feature_names_out(input_features=None):
        Get the names of the output features after transformation.
    """

    def __init__(self):
        self.columns_to_drop = []
        self.transformed_columns = []
        self.feature_mapping = {}

    def get_class_name(self):
        return self.__class__.__name__    
    

    def create_login_app(self, X):
        X_transformed = X.copy()

        X_transformed.loc[:, "IND_LOGIN_APP"] = 0
        self.feature_mapping["IND_LOGIN_APP"] = []
        X_transformed.loc[:, "IND_LOGIN_APP_HI_SITE_EXT"] = 0
        self.feature_mapping["IND_LOGIN_APP_HI_SITE_EXT"] = []
        X_transformed.loc[:, "QT_DAYS_FIRST_LOGIN_APP"] = 0
        self.feature_mapping["QT_DAYS_FIRST_LOGIN_APP"] = []
        X_transformed.loc[:, "QT_DAYS_FIRST_LOGIN_APP_HI_SITE_EXT"] = 0
        self.feature_mapping["QT_DAYS_FIRST_LOGIN_APP_HI_SITE_EXT"] = []
        X_transformed.loc[:, "QT_DAYS_LAST_LOGIN_APP"] = 0
        self.feature_mapping["QT_DAYS_LAST_LOGIN_APP"] = []
        X_transformed.loc[:, "QT_DAYS_LAST_LOGIN_APP_HI_SITE_EXT"] = 0
        self.feature_mapping["QT_DAYS_LAST_LOGIN_APP_HI_SITE_EXT"] = []
        
        for c in X.columns:
            if c.startswith("IND_LOGIN_APP_"):
                X_transformed.loc[:, "IND_LOGIN_APP"] = X_transformed[["IND_LOGIN_APP", c]].max(axis=1)
                self.feature_mapping["IND_LOGIN_APP"].append((c, self.get_class_name()))
            elif c.startswith("IND_LOGIN"):
                X_transformed.loc[:, "IND_LOGIN_APP_HI_SITE_EXT"] = X_transformed[["IND_LOGIN_APP_HI_SITE_EXT", c]].max(axis=1)
                self.feature_mapping["IND_LOGIN_APP_HI_SITE_EXT"].append((c, self.get_class_name()))                
            elif c.startswith("QT_DAYS_LAST_LOGIN_APP_"):
                X_transformed.loc[:, "QT_DAYS_LAST_LOGIN_APP"] = X_transformed[["QT_DAYS_LAST_LOGIN_APP", c]].min(axis=1)
                self.feature_mapping["QT_DAYS_LAST_LOGIN_APP"].append((c, self.get_class_name()))
                X_transformed.loc[:, "QT_DAYS_FIRST_LOGIN_APP"] = X_transformed[["QT_DAYS_FIRST_LOGIN_APP", c]].max(axis=1)
                self.feature_mapping["QT_DAYS_FIRST_LOGIN_APP"].append((c, self.get_class_name()))
            elif c.startswith("QT_DAYS_LAST_LOGIN"):
                X_transformed.loc[:, "QT_DAYS_LAST_LOGIN_APP_HI_SITE_EXT"] = X_transformed[["QT_DAYS_LAST_LOGIN_APP_HI_SITE_EXT", c]].min(axis=1)
                self.feature_mapping["QT_DAYS_LAST_LOGIN_APP_HI_SITE_EXT"].append((c, self.get_class_name()))
                X_transformed.loc[:, "QT_DAYS_FIRST_LOGIN_APP_HI_SITE_EXT"] = X_transformed[["QT_DAYS_FIRST_LOGIN_APP_HI_SITE_EXT", c]].max(axis=1)
                self.feature_mapping["QT_DAYS_FIRST_LOGIN_APP_HI_SITE_EXT"].append((c, self.get_class_name()))
                
        return X_transformed
    
    def split_cabaz_variables(self,X):
        X_transformed = X.copy()
        for original_col in [
            "DSC_CABAZ_ATUAL",
            "DSC_CABAZ_HOM",
            "DSC_FAM_APL_MAIS_ANTIGA",
            "DSC_FAM_APL_MAIS_RECENTE",
            "DSC_FAMILIA_ULT_COMPRA",
            "DSC_FAMILIA_ULT_ANU_APL",
        ]:
            if original_col in X.columns:
                new_apolices = (
                    X[original_col]
                    .str.get_dummies(sep=" / ")
                    .add_prefix(original_col.replace("DSC_", "FLG_") + "_")
                )
                for c in new_apolices.columns:
                    self.feature_mapping[c] = [(original_col, self.get_class_name())]
                X_transformed = pd.concat([X_transformed, new_apolices], axis=1)
                self.columns_to_drop.append(original_col)
        return X_transformed
    
    def create_seguros_financeiros(self, X):
        X_transformed = X.copy()
        transformation_dict = {}
        for c1 in X.columns:
            if c1.endswith('_CAP'):
                for c2 in X.columns:
                    prefix = c1.rsplit('_',1)[0]
                    if prefix in c2 and c2.endswith('_PPR'):
                        transformation_dict[prefix+'_SF'] = [c1, c2]
                        self.feature_mapping[prefix+'_SF'] = [(c1, self.get_class_name()), (c2, self.get_class_name())]

        for p, c in transformation_dict.items():               
            if p.startswith('DSC_'):
                X_transformed[p] = X[c].apply(
                    lambda row: 'S' if 'S' in row.values else 'X' if all(row[col] == 'X' for col in c) else 'N', axis=1
                )

            elif p.startswith('QTD_MESES_ANT') or p.startswith('VAL_MAX'):
                X_transformed[p] = X[c].max(axis = 1)
            elif p.startswith('QTD_MESES') or p.startswith('VAL_MIN'):
                X_transformed[p] = X[c].min(axis = 1)    
            elif p.startswith('QTD_') or p.startswith('VAL_PCA'):
                X_transformed[p] = X[c].sum(axis = 1)

        return X_transformed
    
    def create_flags_from_qtd_meses(self,X):
        new_cols = {}
        for c in X.columns:
            # Create variable that tells if the client have ever had a particular policy
            # it will replaced the renamed ones
            if c.startswith("QTD_MESES_ULT_COMPRA_"):
                new_cols[c.replace("QTD_MESES_ULT_COMPRA_", "DSC_IND_CHURN_")] = np.where(
                    X[c].isna(), "X", "S"
                )
                self.feature_mapping[c.replace("QTD_MESES_ULT_COMPRA_", "DSC_IND_CHURN_")] = [
                    (c, self.get_class_name())
                ]
                self.columns_to_drop.append(c)
            elif c == "QTD_MESES_ULT_CONT":
                new_col = "FLG_CONT_HIST"
                new_cols[new_col] = (X[c] > 0).astype(int)
                self.feature_mapping[new_col] = [(c, self.get_class_name())]
                self.columns_to_drop.append(c)
            elif c.startswith("QTD_MESES_ULT_ANU"):
                new_col = "FLG_APL_ANU_HIST"
                new_cols[c.replace("_APL", "").replace("QTD_MESES_ULT_ANU", new_col)] = (X[c] > 0).astype(
                    int
                )
                self.feature_mapping[new_col] = [(c, self.get_class_name())]
                self.columns_to_drop.append(c)
            elif c == "QTD_MESES_ULT_UPG_APL":
                new_col = "FLG_UPG_APL_HIST"
                new_cols[new_col] = (X[c] > 0).astype(int)
                self.feature_mapping[new_col] = [(c, self.get_class_name())]
                self.columns_to_drop.append(c)
            elif c == "QTD_MESES_ULT_DOWNGR_APL":
                new_col = "FLG_DNG_APL_HIST"
                new_cols[new_col] = (X[c] > 0).astype(int)
                self.feature_mapping[new_col] = [(c, self.get_class_name())]
                self.columns_to_drop.append(c)
        return new_cols
    

    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.columns_to_drop = []
        X_transformed = X.copy()
        
        for c in X_transformed.columns:
            if c.startswith("DSC_IND_CHURN_"):
                # Change variable name
                X_transformed.rename(columns={c: c.replace("DSC_IND_CHURN_", "DSC_IND_CHURN_24M_")}, inplace=True)
                self.feature_mapping[c.replace("DSC_IND_CHURN_", "DSC_IND_CHURN_24M_")] = [(c, self.get_class_name())]
            elif c.startswith("PERC_") or c.startswith("PCT_"):
                # replace -999 for NaN, because -999 is due set when we have division by 0
                X_transformed[c] = np.where(X_transformed[c] == -999, np.nan, X_transformed[c])
            elif (c.startswith("FLG_") or c.startswith("FLAG_")) and X_transformed[c].dtype in ("float", "int"):
                # Transform cases when 'Flag' variable has a value larger than 1
                X_transformed[c] = np.where((pd.isna(X_transformed[c])) | (X_transformed[c] == 0), X_transformed[c], 1)

        
        
        X_transformed = self.create_login_app(X_transformed)
        
        X_transformed = self.split_cabaz_variables(X_transformed)
        
        X_transformed = self.create_seguros_financeiros(X_transformed)
        
        new_cols = self.create_flags_from_qtd_meses(X_transformed)
        X_transformed = pd.concat([X_transformed, pd.DataFrame(new_cols)], axis=1)
        
    
        if "DSC_NUT2" in X_transformed.columns:
            replacement_dict = {
                "REGIÃO AUTÓNOMA DOS AÇORES": "REGIÕES AUTÓNOMAS DOS AÇORES E DA MADEIRA",
                "REGIÃO AUTÓNOMA DA MADEIRA": "REGIÕES AUTÓNOMAS DOS AÇORES E DA MADEIRA",
            }
            X_transformed["DSC_NUT2"] = X["DSC_NUT2"].replace(replacement_dict)

        if "DSC_ESTADO_CIVIL" in X_transformed.columns:
            replacement_dict = {
                "U-COMMON-LAW MARRIAGE": "C-MARRIED/U-COMMON-LAW MARRIAGE",
                "C-MARRIED": "C-MARRIED/U-COMMON-LAW MARRIAGE",
                "D-DIVORCED": "D-DIVORCED/P-SEPARATED/V-WIDOWED",
                "P-SEPARATED": "D-DIVORCED/P-SEPARATED/V-WIDOWED",
                "V-WIDOWED": "D-DIVORCED/P-SEPARATED/V-WIDOWED",
            }
            X_transformed["DSC_ESTADO_CIVIL"] = X["DSC_ESTADO_CIVIL"].replace(replacement_dict)
            
        X_transformed.drop(columns = self.columns_to_drop, inplace=True)    
        self.transformed_columns = X_transformed.columns.tolist()
        return X_transformed
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.transformed_columns


class FeatureTransformerParRar(BaseEstimator, TransformerMixin):
    """
    A transformer to perform feature engineering on the RAR dataset.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data.

    transform(X):
        Transform the input data by applying feature engineering.

    fit_transform(X, y=None):
        Fit the transformer to the data and transform it.

    get_feature_names_out(input_features=None):
        Get the names of the output features after transformation.
    """

    def __init__(self, month_list, data_lob_path):
        self.month_list = month_list
        self.data_lob_path = data_lob_path
        self.prefix_list_sum,  self.prefix_list_flag = {}, {}
        self.columns_to_drop = []
        self.transformed_columns = []
        self.feature_mapping = {}

    def get_class_name(self):
        return self.__class__.__name__  

    def create_features_from_dates(self, X):
        X_transformed = X.copy()
        ano_mes = X.reset_index('ANO_MES')['ANO_MES'].values
        
        if 'DT_INCLUSAO_SEG' in X_transformed.columns:
            X_transformed.loc[:,'QTD_MESES_INCLUSAO_SEG'] = pd.to_numeric(
                (pd.to_datetime(ano_mes,format="%Y%m") - X['DT_INCLUSAO_SEG']) / np.timedelta64(1, 'M'), #/ 365.25 / 12, 
                errors='coerce').round()#.astype('Int64')
            self.feature_mapping['QTD_MESES_INCLUSAO_SEG'] = [('ANO_MES', self.get_class_name()), ('DT_INCLUSAO_SEG', self.get_class_name())]
            self.columns_to_drop.append('DT_INCLUSAO_SEG')
        if 'DT_EXCLUSAO_SEG' in X_transformed.columns:
            X_transformed.loc[:,'QTD_MESES_EXCLUSAO_SEG'] = pd.to_numeric(
                (pd.to_datetime(ano_mes,format="%Y%m") - X['DT_EXCLUSAO_SEG']) / np.timedelta64(1, 'M'), #/ 365.25 / 12, 
                errors='coerce').round()#.astype('Int64')
            self.feature_mapping['QTD_MESES_EXCLUSAO_SEG'] = [('ANO_MES', self.get_class_name()), ('DT_EXCLUSAO_SEG', self.get_class_name())]
            self.columns_to_drop.append('DT_EXCLUSAO_SEG')
        if 'DT_NASCIMENTO_SEG' in X_transformed.columns:
            X_transformed.loc[:,'QTD_IDADE_SEG'] = pd.to_numeric(
                (pd.to_datetime(ano_mes,format="%Y%m") - X['DT_NASCIMENTO_SEG']) / np.timedelta64(1, 'D') / 365.25, 
                errors='coerce').round()#.astype('Int64')
            self.feature_mapping['QTD_IDADE_SEG'] = [('ANO_MES', self.get_class_name()), ('DT_NASCIMENTO_SEG', self.get_class_name())]
            X_transformed.loc[:,'FLG_IDADE_SEG_ATE5'] = X_transformed['QTD_IDADE_SEG'].apply(lambda x: 1 if pd.notna(x) and x <= 5 else 0)
            self.feature_mapping['FLG_IDADE_SEG_ATE5'] = [('QTD_IDADE_SEG', self.get_class_name())]
            X_transformed.loc[:,'FLG_IDADE_SEG_ATE8'] = X_transformed['QTD_IDADE_SEG'].apply(lambda x: 1 if pd.notna(x) and x <= 8 else 0)
            self.feature_mapping['FLG_IDADE_SEG_ATE8'] = [('QTD_IDADE_SEG', self.get_class_name())]
            X_transformed.loc[:,'FLG_IDADE_SEG_ATE10'] = X_transformed['QTD_IDADE_SEG'].apply(lambda x: 1 if pd.notna(x) and x <= 10 else 0)
            self.feature_mapping['FLG_IDADE_SEG_ATE10'] = [('QTD_IDADE_SEG', self.get_class_name())]
            self.columns_to_drop.append('DT_NASCIMENTO_SEG')

            return X_transformed
    
    def create_flag_carencia_sd(self, X):
        if 'DSC_PRODUTO_SO_APL' in X:
            #Variável para associar cada produto à sua tipologia 
            X['TIPO_PRODUTO'] = X['DSC_PRODUTO_SO_APL'].apply(lambda x: 
                                  'MULTICARE_1' if x in ('MULTICARE 1','MULTICARE CTT 1','MULTICARE EUROBIC 1','MULTICARE SANTÉ 1') else 
                                  'MULTICARE_2' if x in ('MULTICARE 2','MULTICARE CTT 2','MULTICARE EUROBIC 2','MULTICARE SANTÉ 2') else 
                                  'MULTICARE_3' if x in ('MULTICARE 3','MULTICARE CTT 3','MULTICARE EUROBIC 3','MULTICARE SANTÉ 3') else 
                                  'PROTECAO_VITAL' if x in ('MULTICARE PROTEÇÃO VITAL','MULTICARE CTT PROTEÇÃO VITAL','MULTICARE EUROBIC PROTEÇÃO VITAL') else None)
            self.feature_mapping['TIPO_PRODUTO'] = [('DSC_PRODUTO_SO_APL', self.get_class_name())]

            periodos_carencia = pd.read_excel(self.data_lob_path / 'Periodos_Carência.xlsx')

            periodos_carencia = periodos_carencia.drop(columns = 'LONG_DESC').set_index('SHORT_DESC').T.add_prefix('COB_').reset_index().rename( columns = {'index':'TIPO_PRODUTO'})

            X_transformed = X.merge(periodos_carencia, on='TIPO_PRODUTO', how='left')
            # Reset the index of X to its original index
            X_transformed.set_index(X.index, inplace=True)

            if 'QTD_MATURIDADE_APL' in X:
                for c in X.columns:
                    if c.startswith('COB_'):
                        X['FLG_CARENCIA_' + c] = (X[c] > X['QTD_MATURIDADE_APL']).astype(int)
                        X.drop(columns = c, inplace=True)
                        self.feature_mapping['FLG_CARENCIA_' + c] = [('SHORT_DESC', self.get_class_name()),('TIPO_PRODUTO', self.get_class_name()), ('QTD_MATURIDADE_APL', self.get_class_name())]

        return X
    

    def compute_prefix_list(self, X):
        self.prefix_list_sum,  self.prefix_list_flag = {}, {}
        for c in X.columns:
            c_split = c.rsplit('_M',1)
            if c_split[-1].isdigit(): #Check if the end of the column is a number
                if X[c].dtype in ("float", "int"):
                    if c_split[0] in self.prefix_list_sum.keys():
                        self.prefix_list_sum[c_split[0]].append(int(c_split[-1]))
                    else:
                        self.prefix_list_sum[c_split[0]] = [int(c_split[-1])]
                else:
                    if c_split[0] in self.prefix_list_flag.keys():
                        self.prefix_list_flag[c_split[0]].append(int(c_split[-1]))
                    else:
                        self.prefix_list_flag[c_split[0]] = [int(c_split[-1])]
                        
    def sum_monthly_variables(self, X):
        new_columns = {}  # Dicionário para armazenar as novas colunas

        for prefix, month_cols in self.prefix_list_sum.items():
            if len(month_cols)>1:
                if month_cols == list(range(min(month_cols),max(month_cols)+1)):
                    for m in self.month_list:
                        new_variable = f'{prefix}_U{m}M'
                        list_variables = [f'{prefix}_M{i}' for i in range(m+1)]

                        # Armazenar a nova coluna no dicionário
                        new_columns[new_variable] = X[list_variables].sum(axis=1)

                        self.feature_mapping[new_variable] = [(c, self.get_class_name()) for c in list_variables]

                    # Determinar as colunas a serem removidas
                    self.columns_to_drop += [f'{prefix}_M{i}' for i in month_cols if i not in ([0, 1] + self.month_list)]

                else:
                    missing_months = []
                    for i in range(min(month_cols),max(month_cols)+1):
                        if i not in month_cols:
                            missing_months.append(i)
                    logging.warning(f"WARNING: Column with prefix {prefix} is missing these months: {missing_months}")
                
        return new_columns
    
    def aggregate_flag_monthly_variables(self, X):
        new_columns = {}  # Dicionário para armazenar as novas colunas

        for prefix, month_cols in self.prefix_list_flag.items():
            if len(month_cols)>1:
                if month_cols == list(range(min(month_cols),max(month_cols)+1)):
                    for m in self.month_list:
                        new_variable = f'{prefix}_U{m}M'
                        list_variables = [f'{prefix}_M{i}' for i in range(m+1)]

                        conditions = [
                            X[list_variables].eq('S').any(axis=1),  # If any column has 'S'
                            X[list_variables].eq('X').all(axis=1)   # If all columns are 'X'
                            ]

                        choices = ['S', 'X']

                        # Armazenar a nova coluna no dicionário
                        new_columns[new_variable] = np.select(conditions, choices, default='N')
                        self.feature_mapping[new_variable] = [(c, self.get_class_name()) for c in list_variables]

                    # Determinar as colunas a serem removidas
                    self.columns_to_drop += [f'{prefix}_M{i}' for i in month_cols if i not in ([0, 1] + self.month_list)]

                else:
                    missing_months = []
                    for i in range(min(month_cols),max(month_cols)+1):
                        if i not in month_cols:
                            missing_months.append(i)
                    logging.warning(f"WARNING: Column with prefix {prefix} is missing these months: {missing_months}")
                
        return new_columns   
    
    def ratio_VAL_PG_VAL_APR(self, X):
        new_columns = {}  # Dicionário para armazenar as novas colunas

        for c1 in X.columns:
            if c1.startswith('VAL_PG'):
                for c2 in X.columns:
                    if c1.replace('PG', 'APR') == c2:
                        new_variable = f'RC_VAL_PG_APR{c1.replace("VAL_PG", "")}'
                        new_columns[new_variable] = X[c1] / X[c2]
                        self.feature_mapping[new_variable] = [(c1, self.get_class_name()), (c2, self.get_class_name())]

        return new_columns

    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        self.columns_to_drop = []
        
        X_transformed = self.create_features_from_dates(X)
            
        if 'VAL_PCA_APL' in X_transformed.columns and 'QTD_SEGURADOS_APL_SD_VIG' in X_transformed.columns:
            #Rácio entre o PCA da apólice e o número de pessoas seguras
            X_transformed['RC_PCA_SEGURADOS_VIG'] = X_transformed['VAL_PCA_APL'] / X_transformed['QTD_SEGURADOS_APL_SD_VIG']
            self.feature_mapping['RC_PCA_SEGURADOS_VIG'] = [('VAL_PCA_APL', self.get_class_name()), ('QTD_SEGURADOS_APL_SD_VIG', self.get_class_name())]
        if 'VAL_PCA_APL_M12' in X_transformed.columns and 'QTD_SEGURADOS_APL_SD_VIG_M12' in X_transformed.columns:
            #Rácio entre o PCA da apólice e o número de pessoas seguras
            X_transformed['RC_PCA_SEGURADOS_VIG_M12'] = X_transformed['VAL_PCA_APL_M12'] / X_transformed['QTD_SEGURADOS_APL_SD_VIG_M12']
            self.feature_mapping['RC_PCA_SEGURADOS_VIG_M12'] = [('VAL_PCA_APL_M12', self.get_class_name()), ('QTD_SEGURADOS_APL_SD_VIG_M12', self.get_class_name())]
            #Variação entre os rácios de PCA e o número de pessoas seguras, em M e em M12
            X_transformed['DIF_RC_PCA_SEGURADOS_VIG_M_M12'] = X_transformed['RC_PCA_SEGURADOS_VIG'].fillna(0) - X_transformed['RC_PCA_SEGURADOS_VIG_M12'].fillna(0)
            self.feature_mapping['DIF_RC_PCA_SEGURADOS_VIG_M_M12'] = [('RC_PCA_SEGURADOS_VIG', self.get_class_name()), ('RC_PCA_SEGURADOS_VIG_M12', self.get_class_name())]
            
        
        X_transformed = self.create_flag_carencia_sd(X_transformed)
        self.compute_prefix_list(X_transformed)
        new_columns = self.sum_monthly_variables(X_transformed)
        new_columns.update(self.aggregate_flag_monthly_variables(X_transformed))
        new_columns.update(self.ratio_VAL_PG_VAL_APR(X_transformed))
        X_transformed = pd.concat([X_transformed, pd.DataFrame(new_columns)], axis=1)
        X_transformed.drop(columns = self.columns_to_drop, axis=1, inplace=True)    
        self.transformed_columns = X_transformed.columns.tolist()
        return X_transformed
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.transformed_columns


class ColumnSelectorTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer to select relevant columns from the dataset.

    Parameters:
    -----------
    columns_to_drop : list, default=[]
        List of column names to drop from the dataset.

    Attributes:
    -----------
    columns_to_drop_unique : set
        Set of unique column names to be dropped from the dataset.
    columns_to_keep : list
        List of column names to keep in the output data.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data.

    transform(X):
        Transform the input data by selecting relevant columns.

    fit_transform(X, y=None):
        Fit the transformer to the data and transform it.

    get_feature_names_out(input_features=None):
        Get the names of the output features after transformation.
    """

    def __init__(self, columns_to_drop = [], missing_threshold = 1):
        self.missing_threshold = missing_threshold
        self.columns_to_drop = columns_to_drop
        self.columns_to_drop_unique = set()
        self.columns_to_keep = []

    def fit(self, X, y=None):

        columns_to_drop = []
        pd.options.mode.use_inf_as_na = True
        X_na = X.isna().sum()/len(X)
        columns_to_drop += X_na[X_na>=self.missing_threshold].index.tolist()

        col_combinations = defaultdict(list)

        # If a column starts or ends with COD it checks if there is another column with the same name
        # but without the 'COD' and removes the ones that start with 'COD'.
        for col in X.columns:
            if col.startswith("COD_"):
                for c in X.columns:
                    if col[4:] in c and not c.startswith("COD_"):
                        col_combinations[c].append(col)
            elif col.endswith("_COD"):
                for c in X.columns:
                    if col[:4] in c and not c.endswith("_COD"):
                        col_combinations[c].append(col)

        for col, combinations in col_combinations.items():
            columns_to_drop.extend(combinations)

        irrelevant_columns = [
            "OD_BGRI_COD_2011",
            "OD_BGRI_COD_2021",
            "IND_BGRI",
            "QT_POLICY_HOLDER",
            "QT_POLICY_INSURED", #qtd apólices que o cliente é tomador (já existe)
            "QT_POLICY_HOLDER_INSURED",
            "VAL_CUSTO_SIN_EX_ABR", #valor dos custos de sinistro em aberto
            "VAL_CUSTO_SIN_EX_FEC", #valor dos custos de sinistro fechados
            "DSC_NUT3_REV1",
            "DSC_HABILITACOES",
            "DSC_SITUACAO_PROF",
            "DSC_PROFISSAO",
            #"DSC_DISTRITO",
            "DSC_CONCELHO",
            "OD_STANDARDIZED_COUNTY",
            "DSC_FREGUESIA",
            "OD_STANDARDIZED_PARISH",
            "DICOFRE", #Código agregado distrito concelho freguesia
            "DICO", #Código agregado distrito concelho
            "QTD_LOJAS_CTT_DICOFRE", #numero de lojas ctt por distrito concelho freguesia
            "QTD_AGC_CGD_DICOFRE",#numero de agências cgds por distrito concelho freguesia
            "COD_SEGMENTO_CLIENTE_MKT", #segmentação antiga de empresas
        ]
        columns_to_drop += irrelevant_columns

        for col in X.columns:
            if "DSC_I_ZONA_SISM_" in col:
                columns_to_drop.append(col)
            else:
                for h in ["50", "60", "70", "80", "90"]:
                    if col == f"DSC_NVL{h}_HIER_COM": #mediador preferencial nivel 10
                        columns_to_drop.append(col)

        financial_products_columns = [
            "VAL_PCA_APL_VIG",
            "VAL_PCA_APL_ANU",
            "VAL_PCA_VIG_ENTRE_6M_12M",
            "VAL_PCA_APL_ANU_UMES",
            "VAL_PCA_APL_ANU_U2M",
            "VAL_PCA_APL_ANU_U3M",
            "VAL_PCA_APL_ANU_U6M",
            "VAL_PCA_APL_ANU_U9M",
            "VAL_PCA_APL_ANU_U12M",
            "VAL_PCA_APL_ANU_U36M",
            "VAL_PCA_APL_VIG_MAIS_72M",
            "VAL_PCA_VIG_ENTRE_48_72M",
            "VAL_PCA_VIG_ENTRE_36_48M",
            "VAL_PCA_VIG_ENTRE_24_36M",
            "VAL_PCA_VIG_ENTRE_12_24M",
        ]
        columns_to_drop += financial_products_columns #há variáveis semelhantes a estas, mas sem os produtos financeiros

        for col in X.columns:
            # Removes datetime columns
            if X[col].dtype == "datetime64[ns]":
                columns_to_drop.append(col)
            # Removes columns with a unique value and no missings
            elif X[col].nunique() == 1 and (X[col].value_counts().iloc[0] / len(X)) == 1:
                columns_to_drop.append(col)
            elif col.endswith("MED_PREF"):  # Drop columns ending with "MED_PREF"
                columns_to_drop.append(col)
            elif col.startswith("DSC_NVL"):  
                columns_to_drop.append(col)
            elif col.startswith("FLG_NVL"):  
                columns_to_drop.append(col)
            elif col.endswith("SIN_ABERT"):  # Drop columns ending with SIN_ABERT
                columns_to_drop.append(col)
            elif col.startswith("OD"):  # Drop columns ending with OD
                columns_to_drop.append(col)

        self.columns_to_drop_unique = set(self.columns_to_drop + columns_to_drop)

        return self

    def transform(self, X):
        self.columns_to_keep = [col for col in X.columns if col not in self.columns_to_drop_unique]
        return X[self.columns_to_keep]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.columns_to_keep

class DataAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, id_column):
        self.id_column = id_column
        self.num_cols = []
        self.flg_cols = []
        self.cat_cols = []
        self.agg_dict = {}
        self.feature_mapping = {}

    def get_class_name(self):
        return self.__class__.__name__  

    def fit(self, X, y=None):
        """
        Identifies numeric, flag, and categorical columns from the dataframe.
        Does nothing else in this case but is useful for pipeline compatibility.
        """
        return self

    def transform(self, X):
        """
        Transforms the dataframe by applying the required aggregations and column manipulations.
        """
        self._categorize_columns(X)
        agg_df = pd.get_dummies(X[self.cat_cols], columns=self.cat_cols, dtype=int)
        
        for cat_col in self.cat_cols:
            transformed_cols = [col for col in agg_df.columns if col.startswith(cat_col)]
            for transformed_col in transformed_cols:
                self.feature_mapping[transformed_col] = [(cat_col,self.get_class_name())]
                
                
        self._prefix_flag_columns(agg_df)
        self._create_agg_dict(agg_df)

        original_columns = X.columns.tolist() + agg_df.columns.tolist()
        # Perform the aggregation
        X_transformed = pd.concat([X,agg_df] ,axis=1).groupby(X.index.names).agg(self.agg_dict)

        # Clean up column names
        X_transformed.columns = ['_'.join(col).strip() for col in X_transformed.columns]
    
        # Map the newly created aggregated columns to the original columns
        for col in X_transformed.columns:
            prefix = col.rsplit('_',1)[0]
            self.feature_mapping[col] = [(prefix, self.get_class_name())]

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fits the transformer to the data and immediately transforms it.
        """
        self.fit(X,y)
        return self.transform(X)

    def _categorize_columns(self, X):
        """
        Classifies the columns in the dataframe as numeric, flag, or categorical columns.
        """
        self.num_cols = []
        self.flg_cols = []
        self.cat_cols = []

        for c in X.columns:
            if X[c].dtype in ("float", "int"):
                if not c.startswith('FLG_'):
                    self.num_cols.append(c)
                else:
                    self.flg_cols.append(c)
            else:
                self.cat_cols.append(c)

    def _prefix_flag_columns(self, agg_df: pd.DataFrame):
        """
        Adds the 'FLG_' prefix to flag columns that don't have the 'FLG_' prefix already.
        """
        renaming_dict = {c: 'FLG_' + c.split('_', 1)[1] for c in agg_df.columns if c.split('_', 1)[0] != 'FLG'}
        agg_df.rename(
            columns= renaming_dict,
            inplace=True
        )
        
        self.feature_mapping.update({v:[(k,self.get_class_name())] for k,v in renaming_dict.items()})

    def _create_agg_dict(self, agg_df: pd.DataFrame):
        """
        Creates the aggregation dictionary based on the identified numeric, flag, and dummy columns.
        """
        self.agg_dict = {
            **{col: ['max', 'min', 'sum', 'mean'] for col in self.num_cols},
            **{col: ['max', 'min', 'mean'] for col in (self.flg_cols + agg_df.columns.tolist())}
        }

class AddMissingColumns(BaseEstimator, TransformerMixin):
    """
    A transformer to add missing columns to a dataset.

    Parameters:
    -----------
    columns_to_keep : list
        A list of column names to be kept in the dataset.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data.

    transform(X):
        Transform the input data by adding missing columns.

    fit_transform(X, y=None):
        Fit the transformer to the data and transform it.

    get_feature_names_out(input_features=None):
        Get the names of the output features after transformation.
    """

    def __init__(self, ordinal_columns=[], categorical_columns=[], drop_new_columns=False):
        self.columns_to_keep = []
        self.ordinal_columns = ordinal_columns
        self.categorical_columns = categorical_columns
        self.columns = []
        self.drop_new_columns = drop_new_columns

 
    def fit(self, X, y=None):
        self.columns_to_keep = X.columns.tolist()
        return self

    def transform(self, X):
        missing_columns = [col for col in self.columns_to_keep if col not in X.columns]
        extra_columns = [col for col in X.columns if col not in self.columns_to_keep]
        logging.warning(f'The following columns were not in the training dataset: {extra_columns}')
                                                                                     
                                                                                                   
        X_missing = {}
        # Add missing columns and fill with NaN
        for col in missing_columns:
            if col.startswith("FLG_") and col not in self.ordinal_columns + self.categorical_columns:
                X_missing[col] = 0
            else:
                logging.warning(
                    f"WARNING: Column {col} isn't in the input dataset so will be added and filled with missing values."
                )
                X_missing[col] = np.nan if col not in self.categorical_columns else "nan"

        X = pd.concat([X, pd.DataFrame(X_missing, index=X.index)], axis=1)

        # Replace pd.NA with np.nan
        X.replace({pd.NA: np.nan}, inplace=True)
        # List of columns to convert to float
        int64_columns = X.select_dtypes(include=["Int64"]).columns

        # Convert Int64 columns to float
        X[int64_columns] = X[int64_columns].astype(float)
        
        if self.drop_new_columns:
            X = X[self.columns_to_keep]
        
        self.columns = X.columns.tolist()
        return X

 
    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.columns

class TargetManualEncoder:
    """
    A class for performing target encoding on non-numerical columns, saving the encoding map to an Excel file.
    """
    def __init__(self, folder, version):
        """
        Initializes the target encoder with the folder to save the encoding values as a parameter.

        """
        self.encoder = TargetEncoder(categories='auto', target_type='binary', smooth='auto', cv=5, shuffle=True, random_state=0).set_output(transform='pandas')
        self.columns_to_encode = []
        self.map_encoders = None
        self.folder = folder
        self.version = version
        self.feature_mapping = {}

    def get_class_name(self):
        return self.__class__.__name__  

    def fit(self, X, y):
        """
        Fit the target encoder on non-numerical columns of the dataset X.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target labels for encoding.
        """
        self._filter_non_numerical_columns(X)
        if len(self.columns_to_encode) > 0:
            self.encoder.fit(X[self.columns_to_encode], y)
        self.save_encodings(X)
        return self

    def transform(self, X):
        """
        Transform the non-numerical columns of X using the fitted target encoder.
    
        Args:
            X (pd.DataFrame): Input features.
    
        Returns:
            pd.DataFrame: Transformed dataset with non-numerical columns encoded.
        """
        X_encoded = X.copy()
        if len(self.columns_to_encode) > 0:
            X_encoded[self.columns_to_encode] = self.encoder.transform(X[self.columns_to_encode])
        return X_encoded

    def fit_transform(self, X, y) :
        """
        Fit the target encoder and transform the dataset in one step.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series): Target labels for encoding.

        Returns:
            pd.DataFrame: Transformed dataset with non-numerical columns encoded.
        """
        self.fit(X, y)
        return self.transform(X)

    def _filter_non_numerical_columns(self, X: pd.DataFrame):
        """
        Filters non-numerical columns from the dataset X to be encoded.

        Args:
            X (pd.DataFrame): Input features.
        """
        self.columns_to_encode = X.select_dtypes(exclude=['number']).columns.tolist()
        #self.columns_to_encode = X.select_dtypes(include=['object', 'category']).columns.tolist()

    def save_encodings(self, X):
        """
        Save the encoding mappings to an Excel file.

        Args:
            X (pd.DataFrame): Input features.
            file_path (str): Path to the output Excel file.
        """
        if len(self.columns_to_encode) > 0:
            map_encoders = {}

            # Compile the encoding map
            for i, c in enumerate(self.columns_to_encode):
                map_encoders[c] = dict(zip(self.encoder.categories_[i], self.encoder.encodings_[i]))

            # Convert the encoding map to a DataFrame and restructure
            df_map_encoders = pd.DataFrame.from_dict(map_encoders, orient='index')
            map_encoders_df = df_map_encoders.reset_index().melt(id_vars='index', var_name='column', value_name='value').dropna()
            
            map_encoders_df.rename(columns={'index':'Column', 'column':'Category', 'value':'Enc. Value'},inplace=True)

            # Save to Excel
            map_encoders_df.to_excel(self.folder / f"map_target_encoders_{self.version.replace('.','-')}.xlsx", index=False)


class OrdinalManualEncoder(BaseEstimator, TransformerMixin):
    """
    An improved transformer to perform ordinal encoding based on predefined or automatically detected categories.

    Parameters:
    -----------
    column_categories : dict, default={}
        A dictionary where keys are column names and values are lists of lists of categories.
        If empty, categories will be automatically detected from the data.
    auto_detect : bool, default=False
        If True, automatically detect categories for columns not specified in column_categories.
    missing_value : str, default="INDETERMINADO"
        Value to use for filling missing values.
    handle_unseen : str, default="encode"
        How to handle unseen categories during transform. Options:
        - 'encode': Use special encoding value
        - 'error': Raise an error
        - 'ignore': Keep values as is

    Attributes:
    -----------
    encoder_dict : dict
        Dictionary mapping column names to their respective OrdinalEncoder instances.
    columns : list
        List of column names being encoded.
    feature_mapping : dict
        Dictionary mapping original feature names to encoded feature names.
    """

    def __init__(self, column_categories={}, auto_detect=False, missing_value="INDETERMINADO", handle_unseen="encode"):
        self.encoder_dict = {}
        self.columns = list(column_categories.keys())
        self.column_categories = column_categories
        self.auto_detect = auto_detect
        self.missing_value = missing_value
        self.handle_unseen = handle_unseen
        self.feature_mapping = {}
        self._validate_params()

    def _validate_params(self):
        """Validate init parameters"""
        valid_handle_unseen = ["encode", "error", "ignore"]
        if self.handle_unseen not in valid_handle_unseen:
            raise ValueError(f"handle_unseen must be one of {valid_handle_unseen}, got {self.handle_unseen}")

        # Validate column_categories structure if provided
        for col, categories in self.column_categories.items():
            if not isinstance(categories, list) or not all(isinstance(cat_list, list) for cat_list in categories):
                raise ValueError(f"Categories for column {col} must be a list of lists")

    def _fill_missing_values(self, X):
        """Fill missing values with the predefined missing value"""
        X_filled = X.copy()
        for col in self.columns:
            # Add the missing value to the categories if not already present
            if col in self.column_categories and X_filled[col].isna().any():
                for i, category_list in enumerate(self.column_categories[col]):
                    if self.missing_value not in category_list:
                        self.column_categories[col][i].append(self.missing_value)

            # Fill missing values
            X_filled.loc[:, col] = X_filled.loc[:, col].fillna(self.missing_value)

        return X_filled

    def fit(self, X, y=None):
        """
        Fit the transformer to the data and encode categorical variables.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to fit the transformer.
        y : array-like, optional
            Target values. Not used.

        Returns:
        --------
        self : object
            Returns self.
        """
        # Auto-detect categories for categorical columns if specified
        if self.auto_detect:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col not in self.column_categories:
                    unique_vals = X[col].dropna().unique().tolist()
                    if self.missing_value not in unique_vals:
                        unique_vals.append(self.missing_value)
                    self.column_categories[col] = [unique_vals]

            # Update columns list
            self.columns = list(self.column_categories.keys())

        # Fill missing values
        X_filled = self._fill_missing_values(X)

        # Create and fit encoders for each column
        for col in self.columns:
            try:
                self.encoder_dict[col] = OrdinalEncoder(
                    categories=self.column_categories[col],
                    handle_unknown="use_encoded_value",
                    unknown_value=len(self.column_categories[col][0]),
                )
                self.encoder_dict[col].fit(X_filled[[col]])

                # Create feature mapping
                encoded_values = {
                    f"{col}={category}": idx
                    for idx, category in enumerate(self.column_categories[col][0])
                }
                self.feature_mapping[col] = encoded_values

            except Exception as e:
                warnings.warn(f"Error fitting encoder for column {col}: {str(e)}")
                # Remove problematic column
                self.columns.remove(col)

        return self

    def transform(self, X):
        """
        Transform the input data by applying the ordinal encoding.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to transform.

        Returns:
        --------
        X_filled : pandas DataFrame
            Transformed data with ordinal encoding applied.
        """
        X_filled = X.copy()

        # Only process columns that exist in the input data
        cols_to_process = [col for col in self.columns if col in X_filled.columns]

        # Fill missing values
        for col in cols_to_process:
            X_filled.loc[:, col] = X_filled.loc[:, col].fillna(self.missing_value)

        # Transform each column
        for col in cols_to_process:
            try:
                # Handle potential new categories
                if self.handle_unseen == "error":
                    # Check for unseen categories
                    unseen_cats = set(X_filled[col].unique()) - set(self.column_categories[col][0])
                    if unseen_cats and self.missing_value not in unseen_cats:
                        raise ValueError(f"Found unseen categories in column {col}: {unseen_cats}")

                # Transform the column
                X_filled[col] = self.encoder_dict[col].transform(X_filled[[col]])

            except Exception as e:
                if self.handle_unseen == "error":
                    raise
                warnings.warn(f"Error transforming column {col}: {str(e)}")
                # Keep original values if error and handle_unseen is 'ignore'

        return X_filled

    def fit_transform(self, X, y=None):
        """
        Fit the transformer to the data, encode categorical variables, and transform it.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to fit and transform.
        y : array-like, optional
            Target values. Not used.

        Returns:
        --------
        X_transformed : pandas DataFrame
            Transformed data with ordinal encoding applied.
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """
        Inverse transform the encoded data back to its original format.

        Parameters:
        -----------
        X : pandas DataFrame
            Encoded data to inverse transform.

        Returns:
        --------
        X_inv : pandas DataFrame
            Data with original categorical values.
        """
        X_inv = X.copy()

        # Only process columns that exist in the input data
        cols_to_process = [col for col in self.columns if col in X_inv.columns]

        for col in cols_to_process:
            try:
                X_inv[col] = self.encoder_dict[col].inverse_transform(X_inv[[col]])[:, 0]
            except Exception as e:
                warnings.warn(f"Error inverse transforming column {col}: {str(e)}")
                # Keep encoded values if error

        return X_inv

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        -----------
        deep : bool, default=True
            If True, return the parameters of all sub-objects.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "column_categories": self.column_categories,
            "auto_detect": self.auto_detect,
            "missing_value": self.missing_value,
            "handle_unseen": self.handle_unseen
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters:
        -----------
        **params : dict
            Estimator parameters.

        Returns:
        --------
        self : object
            Estimator instance.
        """
        for param, value in params.items():
            setattr(self, param, value)

        # Update columns list if column_categories has changed
        if "column_categories" in params:
            self.columns = list(self.column_categories.keys())

        return self

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters:
        -----------
        input_features : array-like of str or None, default=None
            Input features. Not used.

        Returns:
        --------
        feature_names_out : ndarray of str
            Output feature names.
        """
        return np.array(self.columns)

class HighCardinalityThresholdEncoder:
    """
    Aggregate high cardinality categories in a pandas DataFrame into 'Other' category.
    
    This encoder aggregates less frequent categories in a DataFrame column based on
    a threshold percentage of total samples and a minimum percentage for each category.
    Categories that do not meet the specified thresholds are labeled as 'Other'.
    """

    def __init__(self, category_columns=None, threshold=0.8, min_percentage=0.05):
        """
        Initialize the HighCardinalityThresholdEncoder with optional configurations.
        
        Parameters:
        category_columns : list, optional
            A list of columns to be considered for encoding. Default is None, and all categorical
            columns will be considered.
        threshold : float, optional
            The threshold percentage to determine which categories to aggregate. Default is 0.80.
        min_percentage : float, optional
            The minimum percentage required for a category to be retained. Categories with lower
            percentages will be aggregated into 'Other'. Default is 0.05.
        """
        self.category_columns = category_columns if category_columns is not None else []
        self.threshold = threshold
        self.min_percentage = min_percentage
        self.retained_category_dict = {}

    def fit(self, df):
        """
        Fit the encoder on the input DataFrame, identifying categories for aggregation.
        
        Parameters:
        df : pd.DataFrame
            The input DataFrame containing high cardinality categories.
        """
        if not self.category_columns:
            self.category_columns = [col for col in df.columns if df[col].dtype in ["object", "category"]]

        for col in self.category_columns.copy():
            col_series = df[col]
            counts = Counter(col_series)
            total_samples = len(col_series)
            unique_categories_count = col_series.nunique()

            if unique_categories_count == 2:
                continue  # Skip processing for columns with exactly two categories

            retained_categories = []
            cumulative_count = 0

            for category, count in counts.most_common():
                if cumulative_count >= self.threshold * total_samples or count / total_samples < self.min_percentage:
                    break
                cumulative_count += count
                retained_categories.append(category)

            self.retained_category_dict[col] = retained_categories

    def transform(self, df):
        """
        Transform the input DataFrame by aggregating high cardinality categories.
        
        Parameters:
        df : pd.DataFrame
            The input DataFrame to be transformed.
        
        Returns:
        pd.DataFrame
            The transformed DataFrame with aggregated categories.
        """
        if not self.retained_category_dict:
            raise ValueError("You must fit the encoder before transforming")

        df_transformed = df.copy()
        for col, categories in self.retained_category_dict.items():
            df_transformed[col] = df_transformed[col].apply(
                lambda x: x if x in categories else 'Other'
            )

        return df_transformed

    def fit_transform(self, df):
        """
        Fit the encoder and transform the input DataFrame in a single step.
        
        Parameters:
        df : pd.DataFrame
            The input DataFrame containing high cardinality categories.
        
        Returns:
        pd.DataFrame
            The transformed DataFrame with aggregated categories.
        """
        self.fit(df)
        return self.transform(df)



class FillMissingAntCliente(BaseEstimator, TransformerMixin):
    """
    A custom transformer to fill missing values in 'QTD_ANOS_ANTIG_CLIENTE' based on means from the training dataset.

    Parameters:
    -----------
    version : str
        The version of the transformer.
    folder : str
        The folder where the file containing means will be saved.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data and calculate means for filling missing values.

    transform(X):
        Transform the input data by filling missing values.

    fit_transform(X, y=None):
        Fit the transformer to the data, calculate means, and transform it.

    get_params(deep=True):
        Get parameters for this estimator.

    set_params(**params):
        Set the parameters of this estimator.

    get_feature_names_out(input_features=None):
        Get the names of the output features after transformation.
    """

    def __init__(self, version, folder):
        self.version = version
        self.folder = folder
        self.means_ant_cliente = pd.Series()
        self.columns = ["QTD_ANOS_ANTIG_CLIENTE"]

    def fit(self, X, y=None):
        X["FLG_ESTADO_CLIENTE"] = X["FLG_ESTADO_CLIENTE"].fillna("INDETERMINADO")
        self.means_ant_cliente = X.groupby("FLG_ESTADO_CLIENTE")["QTD_ANOS_ANTIG_CLIENTE"].mean()
        self.means_ant_cliente.to_csv(
            f"{self.folder}/media_antiguidade_client_{self.version.replace('.','-')}.csv", index=True
        )

    def transform(self, X):
        # X_transformed = X.copy()
        X["FLG_ESTADO_CLIENTE"] = X["FLG_ESTADO_CLIENTE"].fillna("INDETERMINADO")
        X["QTD_ANOS_ANTIG_CLIENTE"] = X.apply(
            lambda row: (
                self.means_ant_cliente[row["FLG_ESTADO_CLIENTE"]]
                if pd.isnull(row["QTD_ANOS_ANTIG_CLIENTE"])
                else row["QTD_ANOS_ANTIG_CLIENTE"]
            ),
            axis=1,
        )
        return X.drop(columns="FLG_ESTADO_CLIENTE")

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"version": self.version, "folder": self.folder}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_feature_names_out(self, input_features=None):
        return self.columns


class InputMedian(BaseEstimator, TransformerMixin):
    """
    An improved transformer to fill missing values with median values from the training dataset.
    Handles both numerical and categorical data, preserves data types, and provides statistics
    on missing values.

    Parameters:
    -----------
    version : str
        The version of the transformer.
    folder : str
        The folder where the file containing medians will be saved.
    categorical_strategy : str, default='most_frequent'
        Strategy for handling missing values in categorical columns.
        Options: 'most_frequent', 'constant'
    categorical_fill_value : any, default='MISSING'
        Value to use when filling missing values in categorical columns if strategy is 'constant'.

    Attributes:
    -----------
    medians_ : dict
        Dictionary containing median values for numerical columns.
    categorical_fill_ : dict
        Dictionary containing fill values for categorical columns.
    missing_stats_ : dict
        Statistics about missing values in the data.
    columns : list
        Names of the output columns.
    """

    def __init__(self, version, folder, categorical_strategy='most_frequent', categorical_fill_value='MISSING'):
        self.version = version
        self.folder = folder
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value
        self.medians_ = {}
        self.categorical_fill_ = {}
        self.missing_stats_ = {}
        self.columns = []
        self.dtypes_ = {}

    def fit(self, X, y=None):
        """
        Fit the transformer to the data and calculate medians for filling missing values.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to fit the transformer.
        y : array-like, optional
            Target values. Not used.

        Returns:
        --------
        self : object
            Returns self.
        """
        # Store data types
        self.dtypes_ = X.dtypes.to_dict()

        # Calculate missing value statistics
        self.missing_stats_ = {
            'total_missing': X.isna().sum().sum(),
            'missing_by_column': X.isna().sum().to_dict(),
            'missing_percentage': (X.isna().sum() / len(X) * 100).to_dict()
        }

        # Calculate medians for numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        self.medians_ = X[numeric_cols].median(axis=0).to_dict()

        # Calculate fill values for categorical columns based on strategy
        categorical_cols = X.select_dtypes(exclude=['number']).columns
        if categorical_cols.any():
            if self.categorical_strategy == 'most_frequent':
                self.categorical_fill_ = {col: X[col].value_counts().index[0] if not X[
                    col].value_counts().empty else self.categorical_fill_value
                                          for col in categorical_cols}
            else:  # constant strategy
                self.categorical_fill_ = {col: self.categorical_fill_value for col in categorical_cols}

        # Save medians and categorical fill values to file
        data_to_save = {
            'medians': self.medians_,
            'categorical_fill': self.categorical_fill_,
            'missing_stats': self.missing_stats_
        }

        try:
            os.makedirs(self.folder, exist_ok=True)
            with open(f"{self.folder}/medians_{self.version.replace('.', '-')}.json", "w") as fp:
                json.dump(data_to_save, fp, default=str)
        except Exception as e:
            warnings.warn(f"Failed to save median values: {str(e)}")

        return self

    def transform(self, X):
        """
        Transform the input data by filling missing values with medians.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to transform.

        Returns:
        --------
        X_transformed : pandas DataFrame
            Transformed data with missing values filled.
        """
        X_transformed = X.copy()

        # Fill missing values for numeric columns
        numeric_cols = [col for col in self.medians_.keys() if col in X_transformed.columns]
        for col in numeric_cols:
            X_transformed[col] = X_transformed[col].fillna(self.medians_[col])

        # Fill missing values for categorical columns
        categorical_cols = [col for col in self.categorical_fill_.keys() if col in X_transformed.columns]
        for col in categorical_cols:
            X_transformed[col] = X_transformed[col].fillna(self.categorical_fill_[col])

        # Restore original data types
        for col, dtype in self.dtypes_.items():
            if col in X_transformed.columns:
                try:
                    X_transformed[col] = X_transformed[col].astype(dtype)
                except Exception:
                    # If conversion fails, keep as is
                    pass

        self.columns = X_transformed.columns.tolist()
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit the transformer to the data, calculate medians, and transform it.

        Parameters:
        -----------
        X : pandas DataFrame
            Input data to fit and transform.
        y : array-like, optional
            Target values. Not used.

        Returns:
        --------
        X_transformed : pandas DataFrame
            Transformed data with missing values filled.
        """
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        -----------
        deep : bool, default=True
            If True, return the parameters of all sub-objects.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "version": self.version,
            "folder": self.folder,
            "categorical_strategy": self.categorical_strategy,
            "categorical_fill_value": self.categorical_fill_value
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters:
        -----------
        **params : dict
            Estimator parameters.

        Returns:
        --------
        self : object
            Estimator instance.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters:
        -----------
        input_features : array-like of str or None, default=None
            Input features. Not used.

        Returns:
        --------
        feature_names_out : ndarray of str
            Output feature names.
        """
        return np.array(self.columns)

    def get_missing_stats(self):
        """
        Get statistics about missing values in the training data.

        Returns:
        --------
        missing_stats_ : dict
            Dictionary containing missing value statistics.
        """
        return self.missing_stats_

class ColumnsAggregator(BaseEstimator, TransformerMixin):
    """
    A custom transformer to aggregate columns based on lobs.

    Parameters:
    -----------
    version : str
        The version of the experiment.
    folder : str
        The folder to save files related to the transformer.
    lobs : list of str, default=['AP', 'AGR', ...]
        List of lines of business (LOBs) to consider for aggregation.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data and aggregate columns.

    transform(X):
        Transform the input data by aggregating columns based on specified categories.

    fit_transform(X, y=None):
        Fit the transformer to the data, aggregate columns, and transform it.

    get_feature_names_out(input_features=None):
        Get the names of the output features after transformation.
    """

    def __init__(
        self,
        version,
        folder,
        lobs=[
            "AP",
            "AGR",
            "ATCO",
            "ATCP",
            "ATED",
            "AUTO",
            "CACA",
            "CAP",
            "EMB",
            "EQP",
            "INC",
            "MCD",
            "MRH",
            "MRN",
            "NVDO",
            "PPR",
            "RCE",
            "RCF",
            "RCG",
            "RCP",
            "SD",
            "VDO",
            "VDR",
        ],
    ):
        self.version = version
        self.folder = folder
        self.lobs = lobs
        self.super_columns = {}
        self.threshold = 0.01
        self.agg_columns = {}
        self.columns = []
        self.feature_mapping = {}

    def get_class_name(self):
        return self.__class__.__name__

    def fit(self, X, y=None):
        self.super_columns = {}
        self.agg_columns = {}
        # Create super columns based on the provided list of categories
        for c in X.columns:
            super_col, category = "_".join(c.split("_")[:-1]), c.split("_")[-1]
            if super_col not in self.super_columns.keys():
                self.super_columns[super_col] = [category]
            else:
                self.super_columns[super_col].append(category)

        # Remove super columns with fewer than two categories or containing unknown elements
        to_drop = []
        for k, v in self.super_columns.items():
            if len(v) > 2:
                set_v = set(v)
                lobs = set(self.lobs)
                common_elements = set_v.intersection(lobs)
                if len(common_elements) < 2:
                    to_drop.append(k)
            #                 else:
            #                     print('Unknown elements:', k, set_v - lobs)
            else:
                to_drop.append(k)

        for k in to_drop:
            del self.super_columns[k]

        # Create aggregated columns for categories with mean below the threshold
        if y is not None:  # and self.threshold is None:
            self.threshold = y.mean()

        for c in X.columns:
            if "_".join(c.split("_")[:-1]) in self.super_columns.keys():
                # Only aggregates columns with few (ratio below the threshold) values different from 0
                if (X[c] != 0).mean() < self.threshold and not (c.endswith("_SFIN") or c.endswith("_S")):
                    new_col = "_".join(c.split("_")[:-1] + ["OTHERS"])
                    if new_col not in self.agg_columns.keys():
                        self.agg_columns[new_col] = [c]
                    else:
                        self.agg_columns[new_col].append(c)

        with open(f"{self.folder}/aggregated_columns_{self.version.replace('.','-')}.json", "w") as fp:
            json.dump(self.agg_columns, fp)

    def transform(self, X):
        others_df = pd.DataFrame()
        to_drop = []
        for new_col, old_columns in self.agg_columns.items():
            if new_col.startswith("FLG_"):
                others_df[new_col] = X[old_columns].max(axis=1)
            else:
                others_df[new_col] = X[old_columns].sum(axis=1)
            self.feature_mapping[new_col] = [(c, self.get_class_name()) for c in old_columns]
        # Concatenate original and aggregated columns, dropping the original ones
        X_transformed = pd.concat([X.reset_index(drop=True), others_df.reset_index(drop=True)], axis=1).drop(
            columns=to_drop
        )
        self.columns = X_transformed.columns
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.columns


class FlagCreation(BaseEstimator, TransformerMixin):
    """
    A custom transformer to create flag columns based on numeric conditions.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data and identify columns to drop.

    transform(X):
        Transform the input data by creating flag columns based on numeric conditions.

    fit_transform(X, y=None):
        Fit the transformer to the data, identify columns to drop, and transform it.

    get_feature_names_out(input_features=None):
        Get the names of the output features after transformation.
    """

    def __init__(self):
        self.transformed_columns = []
        self.columns_to_drop = []
        self.flg_df = {}
        self.feature_mapping = {}

    def get_class_name(self):
        return self.__class__.__name__

    def fit(self, X, y=None):
        self.columns_to_drop = []
        for c in X.columns:
            if c.startswith("QTD_") and "_SLD_" not in c and "_FIDCOINS_" not in c:
                # if the percentage of values equal to 0 is above 50% otherwise we might lose too much information
                if (X[c] == 0).mean() > 0.5:
                    self.columns_to_drop.append(c)

    def transform(self, X):
        self.flg_df = {}
        X_transformed = X.copy()
        for c in self.columns_to_drop:
            if c.replace("QTD_", "FLG_") not in X_transformed.columns:
                self.flg_df[c.replace("QTD_", "FLG_")] = (X[c] > 0).astype(int)
                self.feature_mapping[c.replace("QTD_", "FLG_")] = [(c, self.get_class_name())]

        X_transformed = pd.concat([X_transformed, pd.DataFrame(self.flg_df)], axis=1).drop(columns=self.columns_to_drop)
        self.transformed_columns = X_transformed.columns.tolist()
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.transformed_columns


class BinningTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to perform binning on numerical features.

    Parameters:
    -----------
    version : str
        Version of the experiment.
    folder : str
        Folder path to save binning information.
    q : int, default=7
        Number of bins to create.
    verbose : bool, default=True
        Whether to display warning messages.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data and determine bin edges for numerical features.

    transform(X):
        Transform the input data by binning numerical features.

    fit_transform(X, y=None):
        Fit the transformer to the data, determine bin edges, and transform it.

    get_feature_names_out():
        Get the names of the output features after transformation.

    adjust_bin_edges(column_values, bins):
        Adjust bin edges based on the minimum and maximum values in the transformed data.
    """

    def __init__(self, version, folder, q=7, verbose=True):
        self.version = version
        self.folder = folder
        self.q = q
        self.bins_dict = {}
        self.columns = []
        self.verbose = verbose

    def fit(self, X, y=None):
        self.bins_dict = {}
        for c in X.columns:
            if X[c].dtype in ("float", "int") and X[c].nunique() > self.q:
                message = ""
                col, bins = pd.qcut(X[c], self.q, labels=False, retbins=True, duplicates="drop")
                if len(bins) > self.q:
                    self.bins_dict[c] = bins.astype(float).tolist()
                elif len(bins) > 2:
                    message = (
                        f"[WARNING] - {col.name} does not have enough variance to use {self.q} bins. "
                        f"It will use {len(bins)-1} instead."
                    )
                    self.bins_dict[c] = bins.astype(float).tolist()
                else:
                    message = f"[WARNING] - {col.name} does not have enough variance to use binning. [IGNORING]"
                if self.verbose and message != "":
                    logging.warning(message)
        with open(f"{self.folder}/binning_columns_{self.version.replace('.','-')}.json", "w") as fp:
            json.dump({c: ["{:0.3f}".format(i) for i in x] for c, x in self.bins_dict.items()}, fp)

    def transform(self, X):
        X_transformed = X.copy()
        for c, bins in self.bins_dict.items():
            # bins[0] -= 0.001
            bins = self.adjust_bin_edges(X_transformed[c], bins)
            X_transformed[c] = pd.cut(X_transformed[c], bins, labels=False, duplicates="drop", include_lowest=True)
        self.columns = X_transformed.columns
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self):
        return self.columns

    def adjust_bin_edges(self, column_values, bins):
        min_value = min(column_values)
        max_value = max(column_values)
        if min_value < bins[0]:
            bins = [min_value] + bins
        if max_value > bins[-1]:
            bins = bins + [max_value]
        return bins


class OutlierTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to detect and adjust outliers in numerical features.

    Parameters:
    -----------
    version : str
        Version of the experiment.
    folder : str
        Folder path to save outlier information.
    q : int, default=7
        Number of bins for binning.
    plot : bool, default=False
        Whether to plot histograms with detected outliers.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data and determine outliers for numerical features.

    transform(X):
        Transform the input data by clipping outliers in numerical features.

    fit_transform(X, y=None):
        Fit the transformer to the data, determine outliers, and transform it.

    get_feature_names_out():
        Get the names of the output features after transformation.

    adjusted_chi2(observed_i, expected_i):
        Calculate the adjusted chi-square value.

    format_thousands(x, pos):
        Format numbers in thousands for plotting.

    adjust_bin_edges(column_values, bins):
        Adjust bin edges based on the minimum and maximum values in the test data.
    """

    def __init__(self, version, folder, q=7, plot=False):
        self.version = version
        self.folder = folder
        self.q = q
        self.plot = plot
        self.adjusted_columns = {}
        self.lower_bounds = {}
        self.upper_bounds = {}

    def adjusted_chi2(self, observed_i, expected_i):
        return np.sum((observed_i - expected_i) ** 2 / observed_i)

    def format_thousands(self, x, pos):
        return f"{x/1000:.1f}K"

    def fit(self, X, y=None):
        self.adjusted_columns = {}
        self.lower_bounds = {}
        self.upper_bounds = {}
        for c in X.columns:
            if X[c].dtype in ("float", "int") and X[c].nunique() > self.q and c != "QTD_IDADE":
                value_counts = X[c].value_counts().sort_index()
                values, frequencies = np.array(value_counts.index), np.array(value_counts.values)

                if c.startswith("QTD_") and X[c].min() >= 0:
                    # Fit a Poisson distribution
                    mu = X[c].mean()
                    poisson_pmf = poisson.pmf(values, mu)
                    # Fit an exponential distribution
                    rate = 1 / (np.sum(values * frequencies) / np.sum(frequencies))
                    exponential_pdf = expon.pdf(values, scale=1 / rate)

                    if self.adjusted_chi2(frequencies, poisson_pmf * np.sum(frequencies)) < self.adjusted_chi2(
                        frequencies, exponential_pdf * np.sum(frequencies)
                    ):
                        upper_bound = poisson.ppf(0.99, mu)
                    else:
                        upper_bound = expon.ppf(0.99, scale=1 / rate)

                    self.adjusted_columns[c] = True
                    self.upper_bounds[c] = upper_bound
                    self.lower_bounds[c] = None

                    if self.plot:
                        # Plot the histogram distribution function
                        plt.bar(values, frequencies, label="Data")
                        plt.plot(values, poisson_pmf * np.sum(frequencies), "ro-", label="Poisson Fit")
                        plt.plot(values, exponential_pdf * np.sum(frequencies), "yo--", label="Exponential Fit")
                        plt.axvline(x=upper_bound, color="g", linestyle="--", label="Upper Bound")
                        plt.xlabel(c)
                        plt.ylabel("Frequencies")
                        plt.gca().yaxis.set_major_formatter(
                            FuncFormatter(self.format_thousands)
                        )  # Apply custom y-axis formatter
                        plt.legend()
                        plt.title(c)
                        plt.show()

                elif not c.startswith("PERC_"):
                    q1 = X[c].quantile(0.25)
                    q3 = X[c].quantile(0.75)
                    iqr = q3 - q1
                    threshold = 1.5
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr

                    self.adjusted_columns[c] = True
                    self.lower_bounds[c] = lower_bound
                    self.upper_bounds[c] = upper_bound

                    if self.plot:
                        # Plot the histogram distribution function
                        plt.hist(X[c], bins=20, alpha=0.7, label="Data")
                        plt.axvline(x=lower_bound, color="r", linestyle="--", label="Lower Bound")
                        plt.axvline(x=upper_bound, color="g", linestyle="--", label="Upper Bound")
                        plt.xlabel(c)
                        plt.ylabel("Frequencies")
                        plt.gca().yaxis.set_major_formatter(
                            FuncFormatter(self.format_thousands)
                        )  # Apply custom y-axis formatter
                        plt.legend()
                        plt.title(c)
                        plt.show()
        with open(f"{self.folder}/outliers_{self.version.replace('.','-')}.json", "w") as fp:
            json.dump(
                {key: (self.lower_bounds[key], self.upper_bounds[key]) for key in self.adjusted_columns.keys()}, fp
            )

    def transform(self, X):
        X_transformed = X.copy()
        for c in self.adjusted_columns:
            X_transformed[c] = np.clip(X_transformed[c], self.lower_bounds[c], self.upper_bounds[c])
        self.columns = X_transformed.columns
        return X_transformed

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self):
        return self.columns


class ValueCountsFilter(BaseEstimator, TransformerMixin):
    """
    A custom transformer to filter features based on the proportion of the most frequent value.

    Parameters:
    -----------
    threshold : float, default=0.99
        Threshold for the proportion of the most frequent value in a feature. Features where the
        proportion exceeds this threshold will be filtered out.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data and determine which features to keep based on the threshold.

    transform(X, y=None):
        Transform the input data by keeping only the selected features.

    fit_transform(X, y=None):
        Fit the transformer to the data, determine features to keep, and transform it.

    get_feature_names_out():
        Get the names of the output features after transformation.
    """

    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.columns_keep = []

    def fit(self, X, y=None):
        self.columns_keep = []
        n = len(X)
        if y is not None:  # and self.threshold is None
            self.threshold = 1 - y.mean() / 2

        for c in X.columns:
            if X[c].value_counts().max() / n < self.threshold:
                self.columns_keep.append(c)

    def transform(self, X, y=None):
        return X[self.columns_keep]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names_out(self):
        return self.columns_keep


class TrainingTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer for training data preprocessing.

    Parameters:
    -----------
    version : str
        Version identifier for the experiment.
    folder : str
        Folder path to save transformer-related files.

    Attributes:
    -----------
    training_grid : dict
        Dictionary containing parameters for the training pipeline.
    cur_pipeline : dict
        Dictionary to store the current pipeline steps.
    X_ca : array-like
        Copy of the training data after ColumnsAggregator transformation.
    X_fl : array-like
        Copy of the training data after FlagCreation transformation.
    X_bin : array-like
        Copy of the training data after BinningTransformer transformation.
    X_outl : array-like
        Copy of the training data after OutlierTransformer transformation.

    Methods:
    --------
    set_grid(training_grid):
        Set the training grid parameters for the transformer.

    run_ColumnsAggregator(X, y, n_combinations):
        Execute the ColumnsAggregator transformation on the training data.

    run_FlagCreation(X, y, n_combinations):
        Execute the FlagCreation transformation on the training data.

    run_BinningTransformer(X, y, n_combinations):
        Execute the BinningTransformer transformation on the training data.

    run_OutlierTransformer(X, y, n_combinations):
        Execute the OutlierTransformer transformation on the training data.

    run_ValueCountsFilter(X, y):
        Execute the ValueCountsFilter transformation on the training data.

    output_string():
        Generate an output string indicating the applied transformations.

    output_pipeline():
        Get the current pipeline steps.
    """

    def __init__(self, version, folder):
        self.training_grid = {}
        self.version = version
        self.folder = folder
        self.cur_pipeline = {}
        self.X_ca = []
        self.X_fl = []
        self.X_bin = []
        self.X_outl = []

    def set_grid(self, training_grid):
        self.training_grid = training_grid
        self.column_aggregation = training_grid["column_aggregation"]
        self.flag_creation = training_grid["flag_creation"]
        self.binning = training_grid["binning"]
        self.outlier_cleaning = training_grid["outlier_cleaning"]
        return self

    def run_ColumnsAggregator(self, X, y, n_combinations):
        self.cur_pipeline["ColumnsAggregator"] = None
        if type(self.X_ca) is list:
            self.X_ca = X.copy()
        X = self.X_ca.copy()
        if self.column_aggregation:
            self.cur_pipeline["ColumnsAggregator"] = ColumnsAggregator(version=self.version, folder=self.folder)
            X = self.cur_pipeline["ColumnsAggregator"].fit_transform(X, y)
            logging.info("%s - column aggregator finished.", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        return X

    def run_FlagCreation(self, X, y, n_combinations):
        self.cur_pipeline["FlagCreation"] = None
        if n_combinations == 2:
            if not self.flag_creation and (type(self.X_fl) is list or self.column_aggregation):
                self.X_fl = X.copy()
            elif self.flag_creation:
                X = self.X_fl.copy()
                # self.X_fl = []
        if self.flag_creation:
            self.cur_pipeline["FlagCreation"] = FlagCreation()
            X = self.cur_pipeline["FlagCreation"].fit_transform(X, y)
            logging.info("%s - flag creation finished.", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        return X

    def run_BinningTransformer(self, X, y, n_combinations):
        self.cur_pipeline["Binning"] = None
        if n_combinations == 2:
            if not self.binning and type(self.X_bin) is list:
                self.X_bin = X.copy()
            elif self.binning:
                X = self.X_bin.copy()
                self.X_bin = []
        if self.binning:
            self.cur_pipeline["Binning"] = BinningTransformer(version=self.version, folder=self.folder, verbose=False)
            X = self.cur_pipeline["Binning"].fit_transform(X, y)
            logging.info("%s - binning transformer finished.", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        return X

    def run_OutlierTransformer(self, X, y, n_combinations):
        self.cur_pipeline["OutlierTransformer"] = None
        if n_combinations == 2 and not self.binning:
            if not self.outlier_cleaning and type(self.X_outl) is list:
                self.X_outl = X.copy()
            elif self.outlier_cleaning:
                X = self.X_outl.copy()
                self.X_outl = []
        if self.outlier_cleaning and not self.binning:
            self.cur_pipeline["OutlierTransformer"] = OutlierTransformer(
                version=self.version, folder=self.folder, plot=False
            )
            X = self.cur_pipeline["OutlierTransformer"].fit_transform(X, y)
            logging.info("%s - outlier cleaning finished.", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
        return X

    def run_ValueCountsFilter(self, X, y):
        self.cur_pipeline["ValueCountsFilter"] = ValueCountsFilter()
        X = self.cur_pipeline["ValueCountsFilter"].fit_transform(X, y)
        return X

    def output_string(self):
        name_mapping = {
            "column_aggregation": "colagg",
            "flag_creation": "flgs",
            "binning": "binn",
            "outlier_cleaning": "outlclean",
        }
        out_str = "_".join([name_mapping[k] for k, v in self.training_grid.items() if v is not False])
        return out_str

    def output_pipeline(self):
        return self.cur_pipeline
