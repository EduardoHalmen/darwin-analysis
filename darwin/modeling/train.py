from pathlib import Path

from loguru import logger

# ---- My imports -----
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import typer

from darwin.config import (
    METRICS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEEDS,
    RANDOM_STATE,
    RAW_DATA_DIR,
    SCORES_DIR,
)

# ---------------------

raw_data_path: Path = RAW_DATA_DIR / "data.csv"
preprocessed_data_path: Path = PROCESSED_DATA_DIR / "preprocessed_data.csv"
feature_imp_path: Path = PROCESSED_DATA_DIR / "feature_imp.csv"
anova_path: Path = PROCESSED_DATA_DIR / "anova.csv"
rfe_path: Path = PROCESSED_DATA_DIR / "rfe.csv"
target_path: Path = PROCESSED_DATA_DIR / "target.csv"

app = typer.Typer()


@app.command()
def main(
    model_path: Path = MODELS_DIR / "model.pkl",
):
    logger.info("Evaluating models...")
    # Define the list of datasets the models will be evaluated on
    raw_data = pd.read_csv(raw_data_path).drop(["ID", "class"], axis="columns")

    df_list = [
        raw_data,
        pd.read_csv(feature_imp_path),
        pd.read_csv(anova_path),
        pd.read_csv(rfe_path),
    ]

    name = ["raw", "feature_imp", "anova", "rfe"]

    for df, name in zip(df_list, name):
        score_path: Path = SCORES_DIR / f"{name}_score.csv"

        # Compute the score for each model
        tree_score = evaluate_model(TREE, df)
        knn_score = evaluate_model(KNN, df)
        mlp_score = evaluate_model(MLP, df)

        # Save the scores to a CSV file
        pd.concat([tree_score, mlp_score, knn_score], keys=["tree", "mlp", "knn"]).to_csv(
            score_path
        )
        logger.info(f"Scores saved to {score_path}")

    logger.success("Evaluating complete.")
    # -----------------------------------------


# Defines the models used in the training
# Might be useful to create a dictionary with the models and their parameters

KNN = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="minkowski",
)

TREE = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    criterion="entropy",
    max_depth=5,
)

MLP = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    random_state=RANDOM_STATE,
    learning_rate="adaptive",
    early_stopping=True,
    max_iter=1000,
)

models = {"knn": KNN, "tree": TREE, "mlp": MLP}


def evaluate_model(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate a model using cross-validation
    The model is evaluated using 5-fold cross-validation,
    metrics used to calculate the final score are defined in METRICS
    For each seed defined in RANDOM_SEEDS, the model is trained and evaluated
    The result is returned as a DataFrame with the metrics evaluated of the model in each seed
    The resulting DataFrame has |RANDOM_SEEDS|x|METRICS| dimensions
        model: estimator
            A sklearn estimator with fit() and predict() methods
        df: pd.DataFrame
            The input data
        return: pd.DataFrame
            A DataFrame with the metrics evaluated of the model in each seed
    """
    # Load the data
    X = df
    y = pd.read_csv(target_path)["class"]

    # Results should be e 2D matrix with |METRICS| columns and |RANDOM_SEEDS| rows
    results = {}

    # Loops through each seed
    # This is done to ensure that the results are robust to the randomness of the data
    for seed in RANDOM_SEEDS:
        scores = {}

        # Define the cross-validation strategy
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        # model.random_state = seed

        # Evaluate the model using cross-validation
        for metric in METRICS:
            scores[metric] = cross_val_score(model, X, y, scoring=metric, cv=kf, n_jobs=-1).mean()

        results[seed] = scores

    return pd.DataFrame(results).T


if __name__ == "__main__":
    app()
