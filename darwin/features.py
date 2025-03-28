from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

# ---- My imports -----
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.svm import SVC

from darwin.config import (
    PROCESSED_DATA_DIR, 
    RANDOM_STATE, 
    CRITERION, 
    MAX_DEPTH, 
    CLASS_WEIGHT, 
    FEATURE_NUM
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
    feature_imp = df[select_feature_imp(df, FEATURE_NUM)]
    anova = df[select_anova(df, FEATURE_NUM)]
    rfe = df[select_rfe(df, FEATURE_NUM)]

    # Save the selected features to a new CSV file
    feature_imp.to_csv(PROCESSED_DATA_DIR / "feature_imp.csv", index=False)
    anova.to_csv(PROCESSED_DATA_DIR / "anova.csv", index=False)
    rfe.to_csv(PROCESSED_DATA_DIR / "rfe.csv", index=False)

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
                                max_depth=MAX_DEPTH,
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
    X = df.drop("class", axis='columns')
    y = df["class"]

    # Fit the selector to the data
    selector = SelectKBest(f_classif, k=n)
    selector.fit(X, y)

    # Get the selected features with highest f-value
    selected_indices = np.argsort(selector.scores_)[::-1][:n]
    selected_features = X.columns[selected_indices].tolist()
    
    return selected_features


def select_rfe(df: pd.DataFrame, n: int) -> list[str]:
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
    X = df.drop("class", axis='columns')
    y = df["class"]

    # Fit the selector to the data
    estimator = SVC(kernel="linear")
    selector = RFE(estimator, n_features_to_select=n, step=1)
    selector = selector.fit(X, y)

    selected_features = X.columns[selector.support_].tolist()

    return selected_features


if __name__ == "__main__":
    app()