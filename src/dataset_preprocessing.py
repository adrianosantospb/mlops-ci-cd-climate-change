import logging
from typing import List

import pandas as pd
import hydra
from omegaconf import DictConfig

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset from a CSV file."""
    logger.info("Loading dataset from %s", path)
    return pd.read_csv(path)


def identify_columns_to_drop(df: pd.DataFrame) -> List[str]:
    """
    Return a list of columns to be removed:
      - columns containing 'net_name'
      - predefined irrelevant columns (useless_cols)
      - columns related to agricultural practices (practice_tokens)
    """
    all_features = df.columns.tolist()

    # Explicitly useless columns
    useless_cols = ["info_gew", "info_resul", "interviewtime", "id", "date"]
    # Tokens that identify agricultural practice features
    practice_tokens = [
        "legum", "conc", "add", "lact", "breed", "covman",
        "comp", "drag", "cov", "plow", "solar", "biog", "ecodr"
    ]

    # Columns with 'net_name' plus predefined useless ones
    drop_list = [feat for feat in all_features if "net_name" in feat] + useless_cols

    # Any column whose name contains any of the practice tokens
    practice_related = [
        feat for feat in all_features
        if any(token in feat for token in practice_tokens)
    ]

    drop_list.extend(practice_related)
    logger.info("Dropping features: %s", drop_list)
    return drop_list


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical (object) columns to numeric codes using pd.factorize.
    Each category is mapped to an integer code.
    """
    non_numeric = df.select_dtypes(include=["object"]).columns
    logger.info("Encoding categorical columns: %s", list(non_numeric))

    for col in non_numeric:
        df[col], _ = pd.factorize(df[col])
    return df


def preprocess_dataset(input_path: str, output_path: str) -> None:
    """
    Full preprocessing pipeline:
      1. Load dataset
      2. Drop irrelevant/practice-related columns
      3. Encode categorical columns
      4. Save the processed dataset
    """
    df = load_dataset(input_path)

    logger.info("Dataset contains %d records and %d features before processing.",
                len(df), len(df.columns))

    cols_to_drop = identify_columns_to_drop(df)
    df.drop(columns=cols_to_drop, inplace=True)

    df = encode_categorical_columns(df)

    logger.info("Saving preprocessed dataset to %s", output_path)
    df.to_csv(output_path, index=False)
    logger.info("Preprocessing complete. Final shape: %s", df.shape)


# --------------------------------------------------------------------------- #
# Entry point with Hydra configuration
# --------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.
    Reads file paths from the configuration and executes the preprocessing pipeline.
    """
    preprocess_dataset(
        input_path=cfg.experiment.dataset_file_path,
        output_path=cfg.experiment.dataset_file_preprocessed
    )


if __name__ == "__main__":
    main()
