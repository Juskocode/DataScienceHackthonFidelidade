import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from dateutil.relativedelta import relativedelta
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV  # , cross_val_score

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV


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


def get_RandomFortestClassifier():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(random_state=0, n_jobs=-1)


def get_LogisticRegression():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(random_state=0, n_jobs=-1)


def get_MLPClassifier():
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(random_state=0)


def get_ExtraTreesClassifier():
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(random_state=0, n_jobs=-1)


def get_GradientBoostingClassifier():
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(random_state=0)


def get_GaussianNB():
    from sklearn.naive_bayes import GaussianNB

    return GaussianNB()


def get_LGBMClassifier():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(random_state=0, n_jobs=-1, verbosity=-1)


model_mapping = {
    "RandomForestClassifier": get_RandomFortestClassifier,
    "LogisticRegression": get_LogisticRegression,
    "MLPClassifier": get_MLPClassifier,
    "ExtraTreesClassifier": get_ExtraTreesClassifier,
    "GradientBoostingClassifier": get_GradientBoostingClassifier,
    "GaussianNB": get_GaussianNB,
    "LGBMClassifier": get_LGBMClassifier,
}


def filter_date(df, n_months=12):
    """
    Filter the dataframe based on a specified number of months.

    Parameters:
    -----------
    df : DataFrame
        Input DataFrame containing a column with dates in the format 'YYYYMM'.
    n_months : int, default=12
        Number of months to consider for training.

    Returns:
    --------
    DataFrame
        Filtered DataFrame containing data from the past n_months.
    """
    # Convert the 'ANO_MES' column to datetime
    df["ANO_MES"] = pd.to_datetime(df["ANO_MES"], format="%Y%m", errors="coerce")
    # Get the latest date in the 'ANO_MES' column
    last_day = df["ANO_MES"].max()
    # Calculate the date that is 'n_months' before the latest date
    past_date = last_day - relativedelta(months=n_months)
    # Filter the DataFrame to include only rows from the past 'n_months'
    return df[df["ANO_MES"] > past_date]


def model_training(X, y, alg_dict, path, calibrated_classifier, roc_threshold=0.7):
    """
    Train models using hyperparameter optimization and return models with scores higher than a specified threshold.

    Parameters:
    -----------
    X : DataFrame
        Input dataframe for training.
    y : Series
        Target variable.
    alg_dict : dict
        Dictionary containing models as keys and their respective hyperparameter grids as values.
    roc_threshold : float, default=0.6
        ROC AUC threshold for selecting models.

    Returns:
    --------
    list
        List of tuples containing the best models and their respective scores.
    """

    alg_output_list = []
    for model_name in alg_dict.keys():
        model_class = model_mapping.get(model_name)()


        if calibrated_classifier is True:
            renamed_grid = {"estimator__" + k: v for k, v in alg_dict[model_name].items()}
            estimator = CalibratedClassifierCV(model_class, n_jobs=-1)
        else:
            renamed_grid = alg_dict[model_name]
            estimator = model_class

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=renamed_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        
        grid_search.fit(X, y)
        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logging.info(f"Best cross-validation score for {model_name}: {grid_search.best_score_}")

        if grid_search.best_score_ > roc_threshold:
            alg_output_list.append((grid_search.best_estimator_, grid_search.best_score_))

        
        brier_sigmoid = brier_score_loss(y, grid_search.predict_proba(X)[:, 0])
        logging.info(f"Brier score: {brier_sigmoid:.4f}")

        fracao_positivos, media_prob_previstas = calibration_curve(y,grid_search.predict_proba(X)[:, 0], n_bins=10)
        plt.figure(figsize=(6, 6))
        plt.plot(media_prob_previstas, fracao_positivos, "s-", label="Model With Calibration")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Average predicted probabilities")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Plot")
        plt.legend()
    
        file_name="calibration_plot.png"
        file_path = path / "Graphics" / file_name
        plt.savefig(file_path, dpi=300)
        plt.close()
        logging.info(f"Calibration plot saved in report/Graphics")

    return alg_output_list

def decile_report(actual_result, predictions, output="print"):
    """
    Perform a decile report based on predicted probabilities and actual results.

    Parameters:
    -----------
    actual_result : array-like
        Array-like object containing the actual target values.
    predictions : array-like
        Array-like object containing the predicted probabilities of class 0.
    output : str, default='print'
        Output format for the report. Options: 'print', 'display', 'streamlit', 'return'.

    Returns:
    --------
    DataFrame or None
        Decile report DataFrame if output is 'return', None otherwise.
    """
    res = pd.DataFrame(np.column_stack((predictions, actual_result)), columns=["PR_0", "Target"])
    res["Decile"] = pd.qcut(res["PR_0"].astype(float), 10, labels=False, duplicates="drop") + 1
    # res['Prob_range'] = pd.qcut(res['PR_0'], 10, duplicates='drop')

    crt = pd.crosstab(res.Decile, res.Target).reset_index()
    crt.columns.name = None
    crt.rename(columns={0.0: "Target 0", 1.0: "Target 1"}, inplace=True)
    quantile_ranges = pd.qcut(res["PR_0"].astype(float), 10, duplicates="drop", retbins=True)[1]
    crt["Minimum"] = quantile_ranges[:-1].round(3)
    crt["Maximum"] = quantile_ranges[1:].round(3)
    crt["Cumulative rate (%)"] = round(crt["Target 1"].cumsum(axis=0) * 100 / crt["Target 1"].sum(axis=0), 1)
    crt["Cumulative Lift"] = round(
        crt["Target 1"].cumsum(axis=0)
        / crt[["Target 0", "Target 1"]].sum(axis=1).cumsum(axis=0)
        / actual_result.mean(),
        2,
    )
    crt = crt[["Decile", "Minimum", "Maximum", "Target 0", "Target 1", "Cumulative rate (%)", "Cumulative Lift"]].copy()
    if output == "print":
        print(crt)
    elif output == "display":
        display(crt)
    elif output == "streamlit":
        st.dataframe(crt)
    elif output == "return":
        return crt


def percentile_report(actual_result, predictions, output="print"):
    """
    Perform a percentile report based on predicted probabilities and actual results.

    Parameters:
    -----------
    actual_result : array-like
        Array-like object containing the actual target values.
    predictions : array-like
        Array-like object containing the predicted probabilities of class 0.
    output : str, default='print'
        Output format for the report. Options: 'print', 'display', 'streamlit', 'return'.

    Returns:
    --------
    DataFrame or None
        Percentile report DataFrame if output is 'return', None otherwise.
    """
    res = pd.DataFrame(np.column_stack((predictions, actual_result)), columns=["PR_0", "Target"])

    try:
        res["Percentile"] = pd.qcut(res["PR_0"].astype(float), 100, labels=False, duplicates="raise") + 1
        quantile_ranges = pd.qcut(res["PR_0"].astype(float), 100, duplicates="raise", retbins=True)[1]
    except Exception:
        logging.warning(
            "Warning: Duplicates found in predicted probabilities. Assigning the same percentile rank to duplicates."
        )
        quantile_ranges = np.percentile(predictions, np.linspace(0, 100, num=100))
        # Assign each data point to the respective quantile
        res["Percentile"] = np.digitize(predictions, quantile_ranges)
        quantile_ranges = np.unique(np.append(quantile_ranges, res["PR_0"].max() + 0.000001))

    crt = pd.crosstab(res.Percentile, res.Target).reset_index()
    crt.columns.name = None
    crt.rename(columns={0.0: "Target 0", 1.0: "Target 1"}, inplace=True)
    crt["Minimum"] = quantile_ranges[:-1].round(3)
    crt["Maximum"] = quantile_ranges[1:].round(3)
    crt["Cumulative rate (%)"] = round(crt["Target 1"].cumsum(axis=0) * 100 / crt["Target 1"].sum(axis=0), 1)
    crt["Cumulative Lift"] = round(
        crt["Target 1"].cumsum(axis=0)
        / crt[["Target 0", "Target 1"]].sum(axis=1).cumsum(axis=0)
        / actual_result.mean(),
        2,
    )
    crt = crt[
        ["Percentile", "Minimum", "Maximum", "Target 0", "Target 1", "Cumulative rate (%)", "Cumulative Lift"]
    ].copy()
    if output == "print":
        print(crt.head(20))
    elif output == "display":
        display(crt.head(20))
    elif output == "streamlit":
        st.dataframe(crt)
    elif output == "return":
        return crt


def save_metrics(actual_result, predictions, test_file, folder, version):
    """
    Save decile and percentile reports to an Excel file and
    also the cumulative rate vs percentile and the cumulative lift vs percentile plots to a PDF.

    Parameters:
    -----------
    actual_result : array-like
        Array-like object containing the actual target values.
    predictions : array-like
        Array-like object containing the predicted probabilities of class 0.
    test_file : str
        Name of the test file used in evaluations.
    folder : str
        Path to the folder where the files will be saved.
    version : str
        Version of the model.
    """
    # Function to plot and save both cumulative rate and cumulative lift plots to a PDF
    # and the decile and percentile report into a xlsx file

    roc_score = roc_auc_score(actual_result, 1 - predictions)
    logging.info("Area under the curve ROC: %f", roc_score)
    roc_df = pd.DataFrame({"ROC_AUC": [roc_score]})
    dr = decile_report(actual_result, predictions, output="return")
    pr = percentile_report(actual_result, predictions, output="return")

    # initialze the excel writer
    writer = pd.ExcelWriter(
        f"{folder}/Percentiles_model{version.replace('.','-')}_test-file_{test_file}.xlsx", engine="xlsxwriter"
    )

    # store your dataframes in a  dict, where the key is the sheet name you want
    frames = {"Deciles": dr, "Percentiles": pr, "ROC_AUC": roc_df}

    # now loop thru and put each on a specific sheet
    for sheet, frame in frames.items():
        frame.to_excel(writer, sheet_name=sheet, index=False)
    writer.close()

    # Define the figure and axes for the plots
    fig, axs = plt.subplots(2, figsize=(10, 12))

    # Plot Cumulative Rate vs Percentile
    pr.loc[-1, ["Percentile", "Cumulative rate (%)"]] = 0
    pr.sort_index(inplace=True)
    axs[0].plot(pr["Percentile"], pr["Cumulative rate (%)"], linestyle="-", color="r", label="Cumulative rate (%)")
    # deciles = pr['Percentile'] % 10 == 0 # Gets the deciles form the percentile report
    axs[0].plot(dr["Decile"] * 10, dr["Cumulative rate (%)"], marker="o", linestyle="", color="r")
    axs[0].plot(pr["Percentile"], pr["Percentile"], linestyle="--", color="grey", label="Random")
    axs[0].set_title("Cumulative Rate vs Percentile")
    axs[0].set_xlabel("Percentile")
    axs[0].set_ylabel("Cumulative rate (%)")
    axs[0].set_xticks(np.arange(0, 101, 10))
    axs[0].set_yticks(np.arange(0, 101, 10))
    axs[0].legend()
    axs[0].grid(False)

    # Plot Cumulative Lift vs Percentile
    axs[1].plot(pr["Percentile"], pr["Cumulative Lift"], linestyle="-", color="r", label="Cumulative Lift")
    axs[1].plot(dr["Decile"] * 10, dr["Cumulative Lift"], marker="o", linestyle="", color="r")
    axs[1].plot(pr["Percentile"], np.ones_like(pr["Percentile"]), linestyle="--", color="grey", label="Random")
    axs[1].set_title("Cumulative Lift vs Percentile")
    axs[1].set_xlabel("Percentile")
    axs[1].set_ylabel("Cumulative Lift")
    axs[1].set_xticks(np.arange(0, 101, 10))
    axs[1].legend()
    axs[1].grid(False)

    # Save the plots to a PDF
    with PdfPages(f"{folder}/graphics/metrics_model{version.replace('.','-')}_test-file_{test_file}.pdf") as pdf:
        pdf.savefig(fig)

    logging.info(
        f"The percentiles were saved in {folder.name}/"
        f"Percentiles_model{version.replace('.','-')}_test-file_{test_file}.xlsx "
        f"and the plots in {folder.name}/graphics/metrics_model{version.replace('.','-')}_test-file_{test_file}.pdf"
    )


def save_feature_importances(pipeline, X_train, y_train, folder, version):
    """
    Save feature importances plots to a PDF file.

    Parameters:
    -----------
    pipeline : Pipeline
        Trained pipeline object containing the selected features steps and the model.
    X_train : array-like
        Array-like object containing the training dataset.
    y_train : array-like
        Array-like object containing the training target values.
    folder : str
        Path to the folder where the files will be saved.
    version : str
        Version of the model.
    """
    selected_features_index = list(pipeline.named_steps.keys()).index("ColumnSelector")
    selected_features = pipeline[selected_features_index].transformers_[0][
        -1
    ]  # ColumnTransformer to filter selected columns
    X_train_sel = pipeline[: selected_features_index + 1].transform(X_train)

    # Generate SHAP plots
    explainer = shap.PermutationExplainer(
        pipeline[selected_features_index + 1 :].predict_proba,
        X_train_sel,
        output_names=["TARGET 0", "TARGET 1"],
        seed=1,
        algorithm="auto",
        max_evals=1000,
        feature_names=selected_features,
    )
    # Selecting just target 1
    shap_values = explainer(X_train_sel[y_train == 1].values)

    fig1 = plt.figure(figsize=(16, 10))
    shap.plots.beeswarm(shap_values[:, :, 1], max_display=20, show=False)  # top 20 variables
    plt.xlabel("Impact on the probability of the target 1")
    plt.ylabel("Features")
    plt.title("Impact of each feature value\nin the target variable 1 (top 20)")
    plt.tight_layout()

    #     # Plot 2: Impact of each feature in the target variable 2
    #     fig2 = plt.figure(figsize=(16, 10))

    #     mean_shap_values = shap._explanation.Explanation(
    #         values=np.mean(shap_values.values[:, :, 1], axis=0),
    #         base_values=shap_values.base_values[0, 1],
    #         data=np.mean(shap_values.data, axis=0),
    #         feature_names=selected_features,
    #     )

    #     shap.plots.waterfall(mean_shap_values, max_display=20, show=False)
    #     plt.ylabel("Features")
    #     plt.title("Impact on probability of target 1 for the\nmean values of the top 20 features")
    #     plt.tight_layout()
    #     plt.subplots_adjust(right=0.92)

    # Plot 2: Feature Importance on training data
    import seaborn as sns
    from sklearn.calibration import CalibratedClassifierCV

    ft_imp = []
    if isinstance(pipeline["Model"], CalibratedClassifierCV):
        for i in pipeline["Model"].calibrated_classifiers_:
            try:
                ft_imp.append(i.estimator.feature_importances_)
            except Exception:
                print(pipeline["Model"].estimator.__class__.__name__, "cannot compute feature importances.")
                ft_imp = [[0]]
                break
        feature_importances = (
            np.mean(ft_imp, axis=0) * 100 / np.sum(ft_imp[0])
            if np.sum(ft_imp[0]) != 0
            else [0] * len(selected_features)
        )
    else:
        try:
            ft_imp = pipeline["Model"].feature_importances_
            feature_importances = ft_imp * 100 / sum(ft_imp)
        except Exception:
            print(pipeline["Model"].__class__.__name__, "cannot compute feature importances.")
            [0] * len(selected_features)

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({"Feature": selected_features, "Importance": feature_importances})

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plot the feature importances
    fig2 = plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df, color="#eb252a")

    # Annotate each bar with the importance value
    for index, value in enumerate(importance_df["Importance"]):
        plt.text(value, index, f" {value:.2f} %", va="center")

    # Remove only the x-axis
    plt.gca().axes.get_xaxis().set_visible(False)
    # Also remove the spines
    sns.despine(left=True, bottom=True)
    plt.title("Feature Importances")
    plt.tight_layout()

    # Save both plots into the same PDF file
    feature_importance_file = f"{folder}/graphics/feature_importance_plots_{version.replace('.','-')}.pdf"
    with PdfPages(feature_importance_file) as pdf:
        pdf.savefig(fig1, bbox_inches="tight")
        pdf.savefig(fig2, bbox_inches="tight")
    logging.info(f"The feature importances were saved in {feature_importance_file}")
    # plt.show()
    return fig1, fig2
