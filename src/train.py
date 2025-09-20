import logging
from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

    useless_cols = ["info_gew", "info_resul", "interviewtime", "id", "date"]
    practice_tokens = [
        "legum", "conc", "add", "lact", "breed", "covman",
        "comp", "drag", "cov", "plow", "solar", "biog", "ecodr"
    ]

    drop_list = [feat for feat in all_features if "net_name" in feat] + useless_cols
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


def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
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
    return df


def plot_summary(df: pd.DataFrame, project_name: str, output_path: str = "summary_plot.png") -> None:
    """
    Create a nicer summary plot (example: count by region) with the project title.
    """
    if "region" not in df.columns:
        logger.warning("Column 'region' not found; skipping plot.")
        return

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=1.1)
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x="region", data=df, order=sorted(df["region"].unique()))
    ax.set_title(f"{project_name} - Records by Region", fontsize=16, weight="bold")
    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    logger.info("Summary plot saved to %s", output_path)


# --------------------------------------------------------------------------- #
# Entry point with Hydra configuration
# --------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.
    Reads file paths from the configuration and executes the preprocessing pipeline,
    then generates a summary plot with the project name as the title.
    """
    df = preprocess_dataset(
        input_path=cfg.experiment.dataset_file_path,
        output_path=cfg.experiment.dataset_file_preprocessed
    )

    # Add a beautiful plot using the experiment name as title
    plot_summary(df, project_name=cfg.experiment.name)


if __name__ == "__main__":
    main()
