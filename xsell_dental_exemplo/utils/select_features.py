import json
import logging
from time import gmtime, strftime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

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


def feature_selection(X, y, train_type, version, folder, overwrite, display_df=True):
    """
    Perform feature selection based on various methods including ANOVA F-value, RandomForest, and GradientBoosting.

    Parameters:
    -----------
    X : DataFrame
        Input dataset.
    y : Series
        Target variable.
    train_type : str
        Type of training.
    version : str
        Version identifier for the experiment.
    folder : str
        Folder path to save feature selection related files.
    overwrite : bool, default=True
        Flag to overwrite the order of the features saved.
    display_df : bool, default=True
        Flag to display the top features DataFrame.

    Returns:
    --------
    list
        List of important features selected by the feature selection process.
    """

    key = "_".join([train_type, version])
    filename = "feature_selection.json"
    with open(folder / filename) as file:
        selected_features = json.load(file)

    if key in selected_features.keys() and overwrite is False:
        # print("Features already saved")
        return selected_features[key]
    else:
        #     X_train = X_train.apply(pd.to_numeric)

        imp_features = []

        scaler = StandardScaler().set_output(transform="pandas")
        X = scaler.fit_transform(X)

        #     chi2_score, chi_2_p_value = chi2(X_norm,y)
        f_score, f_p_value = f_classif(X, y)

        #     imp_features.append(dict(zip(X.columns,(chi2_score - chi2_score.mean())/(chi2_score.std()))))
        # XNormed = (chi2_score - chi2_score.mean())/(chi2_score.std())
        imp_features.append(dict(zip(X.columns, (f_score - f_score.mean()) / (f_score.std()))))

        cols_1 = np.array(X.columns)[np.where(f_p_value < 0.1)[0]]
        #     cols_1 = pd.Series(fixed_columns+list(cols_1)).drop_duplicates().tolist()

        #     X_norm = StandardScaler().fit_transform(X[cols_1])
        #     lr=LogisticRegression(random_state=0,solver='saga',penalty='l1',class_weight='balanced')
        #     lr.fit(X_norm,y)

        #     cols_2 = np.array(cols_1)[np.where(np.array(abs(lr.coef_)>0.001)[0])[0]]
        #     cols_2 = list(set(fixed_columns+list(cols_2)))

        rf = RandomForestClassifier(random_state=0, max_depth=10, class_weight=None, n_jobs=-1)
        rf.fit(X[cols_1], y)

        imp_features.append(
            dict(
                zip(
                    cols_1, (rf.feature_importances_ - rf.feature_importances_.mean()) / (rf.feature_importances_.std())
                )
            )
        )
        cols_3 = np.array(cols_1)[np.where(rf.feature_importances_ > 0)[0]]
        #     cols_3 = pd.Series(fixed_columns+list(cols_3)).drop_duplicates().tolist()

        gbc = GradientBoostingClassifier(max_depth=5, max_features="sqrt", tol=0.001, random_state=0)
        gbc.fit(X[cols_3], y)

        imp_features.append(
            dict(
                zip(
                    cols_3,
                    (gbc.feature_importances_ - gbc.feature_importances_.mean()) / (gbc.feature_importances_.std()),
                )
            )
        )
        cols_4 = np.array(cols_3)[np.where(gbc.feature_importances_ > 0)[0]]
        #     cols_4 = pd.Series(fixed_columns+list(cols_4)).drop_duplicates().tolist()

        dict_feat = dict(zip(cols_4, [[] for _ in range(len(cols_4))]))
        for c in cols_4:
            for i in range(len(imp_features)):
                dict_feat[c].append(imp_features[i][c])

        feature_selection_df = pd.DataFrame.from_dict(dict_feat, orient="index")
        feature_selection_df["Avg_Score"] = np.mean(feature_selection_df, axis=1)
        impt_feat = list(feature_selection_df.sort_values(by=["Avg_Score"], ascending=False).index)
        if display_df:
            display(feature_selection_df.sort_values(by=["Avg_Score"], ascending=False).head(10))
        #     important_features = pd.Series(fixed_columns+impt_feat).drop_duplicates().tolist()
        return impt_feat


def corr_analysis(df, important_features, train_type, version, folder, overwrite):
    """
    Perform correlation analysis to remove highly correlated features.

    Parameters:
    -----------
    df : DataFrame
        Input DataFrame.
    important_features : list
        List of important features selected by the feature selection process.
    train_type : str
        Type of training data.
    version : str
        Version identifier for the correlation analysis process.
    folder : str
        Folder path to save correlation analysis related files.

    Returns:
    --------
    list
        List of uncorrelated features after correlation analysis.
    """
    key = "_".join([train_type, version])
    filename = "feature_selection.json"
    with open(folder / filename) as file:
        selected_features = json.load(file)

    if key in selected_features.keys() and overwrite is False:
        # print("Features already saved")
        return selected_features[key]
    else:
        X = df[important_features]
        corr_matrix = X.corr().abs()
        l_feat = []
        for c in corr_matrix.columns:
            # The features with a correlation, in module, above 0.5 with other variable more important will be removed.
            for i in corr_matrix[corr_matrix[c] > 0.5].index:  # MAY NEED ADJUSTMENT
                if c != i and c not in l_feat:
                    l_feat.append(i)

        for feat in set(l_feat):
            corr_matrix = corr_matrix.drop(feat, axis=1)
            corr_matrix = corr_matrix.drop(feat, axis=0)

        #     print('The number of selected variables is',len(corr_matrix.columns))
        #     selected_features = pd.Series(fixed_columns+list(corr_matrix.columns)).drop_duplicates().tolist()
        selected_features[key] = list(corr_matrix.columns)
        with open(folder / filename, "w") as file:
            json.dump(selected_features, file, ensure_ascii=False)
        return list(corr_matrix.columns)


def make_feature_selection(X, y, version, folder, train_config, overwrite, display):
    """
    Perform feature selection and correlation analysis on the dataset.

    Parameters:
    -----------
    X : DataFrame
        Input dataframe.
    y : Series
        Target variable.
    version : str
        Version identifier for the experiment.
    folder : str
        Folder path to save feature selection related files.
    train_config : str
        Configuration of the training data.

    Returns:
    --------
    list
        List of uncorrelated features after feature selection and correlation analysis.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO

    important_features = feature_selection(X, y, train_config, version, folder, overwrite, display_df=display)
    logging.info("%s - feature selection finished.", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    uncorrelated_features = corr_analysis(X, important_features, train_config, version, folder, overwrite)
    logging.info("%s - correlation selection finished.", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    return uncorrelated_features
