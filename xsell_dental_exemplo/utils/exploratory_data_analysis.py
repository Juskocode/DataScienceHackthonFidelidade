import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import (SelectKBest, chi2, f_classif,
                                       f_regression, mutual_info_classif)

def col_group_by(data_frame: pd.DataFrame, col_name: str):
    """
    Groups and counts occurrences of unique values in a specified column of a data_frame.

    This function takes a data_frame and a column name as input, groups the data_frame by the values
    in the specified column, and then calculates the counts and percentages of each unique value.

    Args:
        data_frame (pandas.DataFrame): The data_frame containing the data to be analyzed.
        col (str): The column name based on which the grouping and counting will be performed.

    Returns:
        pandas.DataFrame: A DataFrame containing the unique values in the specified column, their counts,
        and the corresponding percentage of total counts in descending order.

    """

    grouped = data_frame.groupby(col_name).size().reset_index(name="counts")
    grouped["percentage(%)"] = (grouped["counts"] / grouped["counts"].sum() * 100).round(2)
    return grouped.sort_values(by="percentage(%)", ascending=False)

def missing_reporter(data_frame: pd.DataFrame, use_inf_as_na: bool = True):
    """
    Generate a report on missing values in a DataFrame.

    This function analyzes a given DataFrame and provides a summary of missing values for each column.
    It calculates the total number of missing values, the percentage of missing values, and the data type
    of the columns with missing values.

    Args:
        data_frame (pandas.DataFrame): The DataFrame to be analyzed for missing values.
        use_inf_as_na (bool, optional): Whether to consider infinity values as missing values. Default is True.

    Returns:
        pandas.DataFrame: A DataFrame containing information about missing values for each column.
            - "Nmissings": The total number of missing values in each column.
            - "Pmissings": The percentage of missing values in each column.
            - "ColType": The data type of each column with missing values.
    """

    pd.options.mode.use_inf_as_na = use_inf_as_na
    data_frame_na = data_frame.isna().sum()
    features_na = data_frame_na[data_frame_na > 0]
    data_frame_na = pd.DataFrame.from_dict(
        {
            "# Missings": features_na,
            "% Missings": (features_na.divide(data_frame.shape[0]) * 100).round(2),
            "ColType": data_frame[features_na.index].dtypes,
        }
    )

    data_frame_na = data_frame_na.sort_values(by="# Missings", ascending=False)
    data_frame_na.index.name = 'Variable'

    return data_frame_na

def multivariate_exploration(data_frame: pd.DataFrame):
    """
    Performs multivariate exploration on a pandas DataFrame.
    The goal is to analyze the relationship between each feature in the DataFrame and the target column
    The function provides insights into the distribution of categories for categorical features and the central
        tendency for numeric features concerning the binary target variable.

    Args:
    data_frame (pd.DataFrame): The input DataFrame containing the TARGET column.

    Returns:
    pd.DataFrame: DataFrame containing exploration results.
    """

    exploration_results = []

    for col in tqdm(data_frame.columns, desc="Progress"):
        if col == 'TARGET':
            continue

        # Convert object-type columns to category
        if data_frame[col].dtype == "object":
            data_frame[col] = data_frame[col].astype("category")

        if data_frame[col].dtype == "category":
            main_feature_name = col
            categories = data_frame[col].cat.categories

            category_counts = data_frame.groupby(['TARGET', col]).size().unstack(fill_value=0)
            category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0)

            # Create a MultiIndex column with main feature and sub-feature
            category_columns = pd.MultiIndex.from_product([[main_feature_name], categories], names=["Main", "Sub"])

            # Rename columns to reflect the categories
            category_percentages.columns = category_columns

            exploration_results.append(category_percentages)

        else:
            if data_frame[col].dtype == "object":
                # Handle non-numeric columns (e.g., strings)
                col_mode = data_frame.groupby('TARGET')[col].apply(lambda x: x.mode().iloc[0])
                exploration_results.append(col_mode.rename(f"{col}_mode"))
            else:
                # Handle numeric columns
                numeric_stats = data_frame.groupby('TARGET')[col].mean()
                exploration_results.append(numeric_stats.rename(f"{col}_mean"))

    exploration_data_frame = pd.concat(exploration_results, axis=1).T

    exploration_data_frame.rename(columns={0: 'TARGET_0', 1: 'TARGET_1'}, inplace=True)

    exploration_data_frame.columns.name = None

    exploration_data_frame.index.name = 'Variable'

    exploration_data_frame['TARGET_0'] = exploration_data_frame['TARGET_0'].apply(lambda x: round(x, 3) if not (pd.isna(x) or isinstance(x, pd.Timestamp)) else np.nan)
    exploration_data_frame['TARGET_1'] = exploration_data_frame['TARGET_1'].apply(lambda x: round(x, 3) if not (pd.isna(x) or isinstance(x, pd.Timestamp)) else np.nan)

    exploration_data_frame['Absolute difference'] = (exploration_data_frame['TARGET_0']-exploration_data_frame['TARGET_1']).abs()
    # #ignore differences that are smaller than the TARGET ratio
    exploration_data_frame['Difference ratio'] = np.where(exploration_data_frame['Absolute difference'] > (data_frame['TARGET'].mean()),exploration_data_frame['Absolute difference']/(exploration_data_frame.abs().max(axis=1)),np.nan)
    exploration_data_frame['Difference ratio'] = exploration_data_frame['Difference ratio'].apply(lambda x: round(x, 3) if not (pd.isna(x) or isinstance(x, pd.Timestamp)) else np.nan)
    
    return exploration_data_frame.sort_values(by = ['Difference ratio','Absolute difference'], ascending=False)

def summary_numeric_stats(data_frame: pd.DataFrame):
    """
    Generate summary statistics for numeric features in a DataFrame.

    This function calculates basic summary statistics for the numeric columns (excluding categorical columns)
    in the given DataFrame. It provides information such as the mean, standard deviation, minimum, 25th percentile,
    median (50th percentile), 75th percentile, and maximum values.

    Args:
        data_frame (pandas.DataFrame): The DataFrame containing the numeric features to be summarized.

    Returns:
        pandas.DataFrame: A DataFrame containing summary statistics for each numeric column.
            Columns:
            - "mean": Mean value of each numeric column.
            - "std": Standard deviation of each numeric column.
            - "min": Minimum value of each numeric column.
            - "25%": 25th percentile (first quartile) value of each numeric column.
            - "50%": Median (50th percentile) value of each numeric column.
            - "75%": 75th percentile (third quartile) value of each numeric column.
            - "max": Maximum value of each numeric column.
    """
    pd.set_option("display.max_rows", None)
    pd.options.display.float_format = "{:.3f}".format
    num_features = data_frame.select_dtypes(exclude=["category", "object"]).columns
    data_frame_num_stats = pd.DataFrame(data_frame[num_features].describe().loc["mean":, :].round(2).T)
    return data_frame_num_stats

def get_feature_correlations(
    data_frame: pd.DataFrame,
    target_col: pd.Series,
    k: int = 5,
    method: str = "spearmanr",
    score_func: str = "f_regression",
):
    """
    Calculate the correlations between features and a target variable using SelectKBest.

    Parameters:
    - data_frame: Pandas DataFrame containing the dataset.
    - target_col: Name of the target column (a Pandas Series or a NumPy array).
    - k: Number of top features to select.
    - method (`str`): Method of correlation: spearmanr or pearsonr. Default is spearmanr
    - score_func (`str`): Score function to use: f_regression,chi2,f_classif,mutual_info_classif.
            Default is f_regression.

    Returns:
    - DataFrame with columns 'Feature' and 'Correlation' containing the top-k features
      and their correlations with the target variable.
    """
    data_frame = data_frame.select_dtypes(include=[np.number])

    data_frame = data_frame[data_frame.columns[~data_frame.isnull().any()]]

    num_cols_len = len(data_frame)

    if num_cols_len < k:
        print("[WARNING] - The number of numerical cols is less than the given k.")
        print("[INFO] - Setting k to", num_cols_len)
        k = num_cols_len

    # Select top-k features using SelectKBest
    methods = {
        "chi2": chi2,
        "f_classif": f_classif,
        "mutual_info_classif": mutual_info_classif,
        "f_regression": f_regression,
    }
    selector = SelectKBest(score_func=methods[score_func], k=k)
    selector.fit_transform(data_frame, target_col)

    # Get the indices of the selected features
    selected_indices = selector.get_support(indices=True)
    selected_features = data_frame.columns[selected_indices]

    if method == "spearmanr":
        # Calculate the correlations with the target variable
        correlations = [spearmanr(target_col, data_frame[feature])[0] for feature in selected_features]
    else:
        correlations = [pearsonr(target_col, data_frame[feature])[0] for feature in selected_features]

    # Create a DataFrame to store the results
    result_data_frame = pd.DataFrame({"Feature": selected_features, "Correlation": correlations})
    result_data_frame = result_data_frame.sort_values(by="Correlation", ascending=False)

    return result_data_frame

def categorical_effectiveness(df, TARGET_col, min_count=0):
    """
    Fit and transform the data for multiple categorical columns simultaneously.

    Parameters:
    - df (pd.DataFrame): The input data containing the categorical columns without TARGET column.
    - TARGET_col (pd.Series): The TARGET column.
    - min_count: Minimum number of records per category to be saved.

    Returns:
    pd.DataFrame: Transformed DataFrame with effectiveness metrics for all categorical columns with at least 10 values by category.
    """

    df.loc[:, [c for c in df.columns if df[c].nunique() == 2]] = df.loc[:, [c for c in df.columns if df[c].nunique() == 2]].astype('category')

    df = pd.concat([pd.DataFrame(df), pd.Series(TARGET_col, name="TARGET")], axis=1)
    TARGET_col = "TARGET"
    columns = []

    categ_cols = df.select_dtypes(include=["object", "category"]).columns.to_list()

    # Replace missing values with "missing"
    df[categ_cols] = df[categ_cols].fillna("Missing")

    len_df = len(df)
    TARGET_total = df[TARGET_col].sum()
    TARGET_percentage = df[TARGET_col].mean() * 100

    for col in categ_cols:
        if df[col].dtype != "category":
            df[col] = df[col].astype("category")
        df[col] = df[col].cat.as_ordered()

        table_df = (
            df.groupby(col)
            .agg(COUNT=pd.NamedAgg(col, "count"), SUM_TARGET=pd.NamedAgg(TARGET_col, "sum"), Effectiveness=pd.NamedAgg(TARGET_col, "mean"))
            .reset_index()
        )
        table_df.rename(columns={col: "Categories"}, inplace=True)
        table_df["% COUNT"] = table_df["COUNT"] / len_df * 100
        table_df["% Target 1"] = table_df["SUM_TARGET"] / TARGET_total * 100
        table_df["TARGET_percentage"] = TARGET_percentage
        table_df["Effectiveness"] *= 100
        table_df["Conclusion"] = table_df.apply(
            lambda row: "Maior propensão" if row["Effectiveness"] > row["TARGET_percentage"] + 0.001 else "", axis=1
        )
        table_df["Variable"] = col

        columns.append(table_df)

    result = pd.concat(columns)
    result["Effectiveness"] = result["Effectiveness"].apply(lambda x: round(x, 3) if not pd.isna(x) else np.nan)
    result.rename(columns={"SUM_TARGET": "# Target 1", "Effectiveness": "Effectiveness (%)"}, inplace=True)
    return result[result['COUNT'] > min_count]


def numerical_effectivness(df,TARGET_col):
    
    columns = []
    num_cols = df.select_dtypes(exclude=["object", "category", "datetime64"]).columns.to_list()
    df = pd.concat([pd.DataFrame(df), pd.Series(TARGET_col, name="TARGET")], axis=1)
    TARGET_col = "TARGET"
    
    len_df = len(df)
    TARGET_total = df[TARGET_col].sum()
    TARGET_percentage = df[TARGET_col].mean() * 100
    
    for col in num_cols:
        if not(df[col].isna().all()) and df[col].nunique() != 2:
            groups, bins = pd.qcut(df[col],q=10, labels=None,retbins=True,duplicates='drop')
            
            if len(bins) == 2:
                argmax = df[col].value_counts().argmax()
                others = [i for i in df[col].unique() if i != argmax]
                groups = pd.Series(np.where(df[col] == argmax, argmax, str([min(others), max(others)])),name = col)
                bins = sorted([argmax]+[min(others), max(others)])
            
            df_group = pd.concat([df[TARGET_col],groups],axis=1)
            agg_dict = {TARGET_col: ['mean', 'count','sum']}
            eff = df_group.groupby(col,as_index=False).agg(**{f'{TARGET_col}_{agg_func}': (TARGET_col, agg_func) for agg_func in agg_dict[TARGET_col]})
            for i in range(len(eff)):
                eff.loc[i,['Lower Limit','Upper Limit']] = bins[i], bins[i+1]
                
            eff["% COUNT"] = eff[f"{TARGET_col}_count"] / len_df *100
            eff["# Target 1"] = eff[f"{TARGET_col}_sum"].astype(int)
            eff["% Target 1"] = eff["# Target 1"] / TARGET_total *100
            eff['Effectiveness (%)'] = (eff[f'{TARGET_col}_mean'] * 100).apply(lambda x: round(x, 2) if not pd.isna(x) else np.nan)
            eff['Conclusion'] = np.where(eff['Effectiveness (%)'] > TARGET_percentage+0.001,"Maior propensão","")
            eff['Variable'] = col
            eff = eff.rename(columns = {col:"Categories", f"{TARGET_col}_count": "COUNT"}).drop(columns=[f'{TARGET_col}_mean', f'{TARGET_col}_sum'])
            columns.append(eff)
        
    result = pd.concat(columns)
    return result

class NumericalEffectivenessComp:
    def __init__(self):
        self.saved_bins = {}  # To store bin boundaries for each numerical column

    def fit(self, df, TARGET_col):
        """
        Fit the model by creating bins for numerical columns and saving them.

        Parameters:
        - df (pd.DataFrame): The input data containing the numerical columns without TARGET column.
        - TARGET_col (pd.Series): The TARGET column.

        Returns:
        pd.DataFrame: Transformed DataFrame with effectiveness metrics for all numerical columns in the training data.
        """
        columns = []
        num_cols = df.select_dtypes(exclude=["object", "category", "datetime64"]).columns.to_list()
        df = pd.concat([pd.DataFrame(df), pd.Series(TARGET_col, name="TARGET")], axis=1)
        TARGET_col = "TARGET"
        
        len_df = len(df)
        TARGET_total = df[TARGET_col].sum()
        TARGET_percentage = df[TARGET_col].mean() * 100

        for col in num_cols:
            if not (df[col].isna().all()) and df[col].nunique() > 2:
                groups, bins = pd.qcut(df[col], q=10, labels=None, retbins=True, duplicates="drop")
                self.saved_bins[col] = bins  # Save bins for this column

                df_group = pd.concat([df[TARGET_col], groups.rename(col)], axis=1)
                agg_dict = {TARGET_col: ["mean", "count", "sum"]}
                eff = df_group.groupby(col, as_index=False).agg(
                    **{f"{TARGET_col}_{agg_func}": (TARGET_col, agg_func) for agg_func in agg_dict[TARGET_col]}
                )
                for i in range(len(eff)):
                    eff.loc[i, ["Lower Limit", "Upper Limit"]] = bins[i], bins[i + 1]

                eff["% COUNT"] = eff[f"{TARGET_col}_count"] / len_df * 100
                eff["# Target 1"] = eff[f"{TARGET_col}_sum"].astype(int)
                eff["% Target 1"] = eff["# Target 1"] / TARGET_total * 100
                eff["Effectiveness (%)"] = (
                    eff[f"{TARGET_col}_mean"] * 100
                ).apply(lambda x: round(x, 2) if not pd.isna(x) else np.nan)
                eff["Conclusion"] = np.where(
                    eff["Effectiveness (%)"] > TARGET_percentage + 0.001, "Maior propensão", ""
                )
                eff["Variable"] = col
                eff = eff.rename(columns={col: "Categories", f"{TARGET_col}_count": "COUNT"}).drop(
                    columns=[f"{TARGET_col}_mean", f"{TARGET_col}_sum"]
                )
                columns.append(eff)

        result = pd.concat(columns)
        return result

    def apply_and_calculate_effectiveness(self, df, TARGET_col):
        """
        Apply the saved bins to test data and compute effectiveness metrics based on training data bins.

        Parameters:
        - df (pd.DataFrame): The input data to transform and calculate effectiveness.
        - TARGET_col (pd.Series): The TARGET column for the test data.

        Returns:
        pd.DataFrame: Effectiveness metrics for the test data, using the bins from the training data.
        """
        columns = []
        df = pd.concat([pd.DataFrame(df), pd.Series(TARGET_col, name="TARGET")], axis=1)
        TARGET_col = "TARGET"

        len_df = len(df)
        TARGET_total = df[TARGET_col].sum()
        TARGET_percentage = df[TARGET_col].mean() * 100

        for col, bins in self.saved_bins.items():
            if col in df.columns:
                # Apply the saved bins to the test data
                groups = pd.cut(df[col], bins=bins, include_lowest=True, labels=False)
                df_group = pd.concat([df[TARGET_col], groups.rename(col)], axis=1)
                agg_dict = {TARGET_col: ["mean", "count", "sum"]}
                eff = df_group.groupby(col, as_index=False).agg(
                    **{f"{TARGET_col}_{agg_func}": (TARGET_col, agg_func) for agg_func in agg_dict[TARGET_col]}
                )
                for i in range(len(eff)):
                    eff.loc[i, ["Lower Limit", "Upper Limit"]] = bins[i], bins[i + 1]

                eff["% COUNT"] = eff[f"{TARGET_col}_count"] / len_df * 100
                eff["# Target 1"] = eff[f"{TARGET_col}_sum"].astype(int)
                eff["% Target 1"] = eff["# Target 1"] / TARGET_total * 100
                eff["Effectiveness (%)"] = (
                    eff[f"{TARGET_col}_mean"] * 100
                ).apply(lambda x: round(x, 2) if not pd.isna(x) else np.nan)
                eff["Conclusion"] = np.where(
                    eff["Effectiveness (%)"] > TARGET_percentage + 0.001, "Maior propensão", ""
                )
                eff["Variable"] = col
                eff = eff.rename(columns={col: "Categories", f"{TARGET_col}_count": "COUNT"}).drop(
                    columns=[f"{TARGET_col}_mean", f"{TARGET_col}_sum"]
                )
                columns.append(eff)

        result = pd.concat(columns)
        return result

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

# Define a function to format y-axis labels
def format_thousands(x, pos):
    return f'{x/1000:.0f}K'

def plot_distribution (df, column,pdf_pages=None, show = False):

    if df[column].dtype in ('object', 'category') or (df[column].dtype in ('float', 'int') and df[column].nunique() <= 10) and not(df[column].isna().all()):
        plt.figure(figsize=(6, 4))
        df[column].value_counts().plot(kind='bar', edgecolor='black', grid=False) 
        plt.title(f'Bar plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))  # Apply custom y-axis formatter
        if pdf_pages:
            pdf_pages.savefig(bbox_inches='tight')  # Save the current figure to the PDF
        if show:
            plt.show() 
        plt.close()
    elif df[column].dtype in ('float', 'int') and df[column].nunique() > 10:
        plt.figure(figsize=(8, 4))
        df[column].hist( edgecolor='black', grid=False) #bins=10,
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_thousands))  # Apply custom y-axis formatter
        if pdf_pages:
            pdf_pages.savefig(bbox_inches='tight')  # Save the current figure to the PDF        
        if show:
            plt.show()
        plt.close()
    else:
        print(f"Column {column} could not be printed")