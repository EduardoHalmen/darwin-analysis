from pathlib import Path

from loguru import logger
import numpy as np

# ---- My imports -----
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import typer

from darwin.config import (
    CLASS_WEIGHT,
    CRITERION,
    FEATURE_NUM,
    MAX_DEPTH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)

# ---------------------

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "preprocessed_data.csv",
):
    logger.info("Generating features from dataset...")
    df = pd.read_csv(input_path)

    # Create a subset of the preprocessed dataset with only the selected features
    logger.info("Selecting features from RandomForest...")
    feature_imp = df[select_feature_imp(df, FEATURE_NUM)]
    logger.info("Selecting features from SelectKBest...")
    anova = df[select_anova(df, FEATURE_NUM)]
    logger.info("Selecting features from RFE using Logistic Regressor...")
    rfe_log = df[select_rfe(df, FEATURE_NUM, 'log')]

    
    logger.info("Selecting features from RFE using Decision Tree...")
    rfe_tree = df[select_rfe(df, FEATURE_NUM, 'tree')]
    # logger.info("Selecting features from RFE using GDB...")
    # rfe_gdb = df[select_rfe(df, FEATURE_NUM, 'gdb')]

    # Save the selected features to a new CSV file
    logger.info("Saving new data to CSV files...")
    feature_imp.to_csv(PROCESSED_DATA_DIR / "feature_imp.csv", index=False)
    anova.to_csv(PROCESSED_DATA_DIR / "anova.csv", index=False)
    rfe_log.to_csv(PROCESSED_DATA_DIR / "rfe.csv", index=False)

    rfe_tree.to_csv(PROCESSED_DATA_DIR / "rfe_tree.csv", index=False)
    # rfe_gdb.to_csv(PROCESSED_DATA_DIR / "rfe_gdb.csv", index=False)

    logger.success("Features generation complete.")
    # -----------------------------------------


def select_feature_imp(df: pd.DataFrame, n: int) -> list[str]:
    """
    Given a DataFrame, returns the n most important features based on the 
    feature_importance_ of a RandomForestClassifier
        df: pd.DataFrame
            DataFrame with the data
        n: int
            Number of features to return
        return: list
            List with the n most important features
    """
    # Splits the target and the features
    X = df.drop("class", axis='columns')
    y = df["class"]
    
    # Train the Forest
    forest = RandomForestClassifier(n_estimators=100,
                                random_state=RANDOM_STATE,
                                criterion=CRITERION,
                                max_depth=7,
                                class_weight=CLASS_WEIGHT)
    forest.fit(X, y)

    # Get the feature importance
    feature_imp = pd.Series(forest.feature_importances_, index=X.columns)
    feature_imp = feature_imp.sort_values(ascending=False)

    selected_features = feature_imp.head(n).index.tolist()

    return selected_features


def select_anova(df: pd.DataFrame, n: int) -> list[str]:
    """
    Given a DataFrame, returns the n most important features based on the
    ANOVA F-Value from SelectKBest
        df: pd.DataFrame
            DataFrame with the data
        n: int
            Number of features to return
        return: list
            List with the n most important features
    """
    # Splits the target and the features
    X = df.drop("class", axis="columns")
    y = df["class"]

    # Fit the selector to the data
    selector = SelectKBest(f_classif, k=n)
    selector.fit(X, y)

    # Get the selected features with highest f-value
    selected_indices = np.argsort(selector.scores_)[::-1][:n]
    selected_features = X.columns[selected_indices].tolist()

    return selected_features


def select_rfe(df: pd.DataFrame, n: int, estim: str='log') -> list[str]:
    """
    Given a DataFrame, returns the n most important features selected
    by Recursive Feature Elimination with a Support Vector Classifier
        df: pd.DataFrame
            DataFrame with the data
        n: int
            Number of features to return
        return: list
            List with the n most important features
    """
    # Splits the target and and features
    X = df.drop("class", axis="columns")
    y = df["class"]

    if estim == 'log':
    # Fit the selector to the data
        estimator = LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight=CLASS_WEIGHT,
        )

    elif estim == 'tree':
        estimator = DecisionTreeClassifier()

    elif estim == 'gdb':
        estimator = GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        )

    selector = RFE(estimator, n_features_to_select=n, step=1)
    selector = selector.fit(X, y)

    selected_features = X.columns[selector.support_].tolist()

    return selected_features

def select_shap(df: pd.DataFrame, n: int) -> list[str]:
    """
    Given a DataFrame, returns the n most important features selected
    by SHAP
        df: pd.DataFrame
            DataFrame with the data
        n: int
            Number of features to return
        return: list
            List with the n most important features
    """
    gdb = GradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.1,
        n_estimators=50,
        max_depth=7,
        criterion='friedman_mse',
        max_features='log2',
        min_samples_split=10,
        random_state=RANDOM_STATE
    )
    
    X = df.drop("class", axis='columns')
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

    gdb.fit(X_train, y_train)

    explainer = shap.TreeExplainer(gdb)

    # Calculate SHAP values for the entire test set
    shap_values = explainer.shap_values(X_train)
    # Calculate the mean absolute SHAP values for each feature
    shap_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": np.abs(shap_values).mean(axis=0)
    })

    # Sort features by importance in descending order
    shap_importance = shap_importance.sort_values(by="importance", ascending=False)

    # Select the top n features
    top_features = shap_importance.head(n)["feature"].tolist()
    return top_features






if __name__ == "__main__":
    app()
