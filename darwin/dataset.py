from pathlib import Path

from loguru import logger

# ---- My imports -----
import pandas as pd
from sklearn.preprocessing import StandardScaler
import typer

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# ---------------------

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "preprocessed_data.csv",
):
    logger.info("Processing dataset...")
    df = pd.read_csv(input_path)

    X = df.drop(["ID", "class"], axis="columns")
    # Remaps the target column to 0 and 1
    y = df["class"].map({"P": True, "H": False})

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.set_output(transform="pandas").fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled["class"] = y

    X_scaled.to_csv(output_path, index=False)

    # Saves the target column for precaution
    y.to_csv(PROCESSED_DATA_DIR / "target.csv", index=False)

    logger.info(f"Processed dataset saved to {output_path}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
