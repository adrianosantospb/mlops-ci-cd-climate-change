import json
import logging
from typing import List, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
    """Return list of irrelevant or practice-related columns to drop."""
    all_features = df.columns.tolist()
    useless_cols = ["info_gew", "info_resul", "interviewtime", "id", "date"]
    practice_tokens = [
        "legum", "conc", "add", "lact", "breed", "covman",
        "comp", "drag", "cov", "plow", "solar", "biog", "ecodr"
    ]
    drop_list = [f for f in all_features if "net_name" in f] + useless_cols
    drop_list.extend([f for f in all_features if any(t in f for t in practice_tokens)])
    logger.info("Dropping features: %s", drop_list)
    return drop_list


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical (object) columns to integer codes."""
    non_numeric = df.select_dtypes(include=["object"]).columns
    logger.info("Encoding categorical columns: %s", list(non_numeric))
    for col in non_numeric:
        df[col], _ = pd.factorize(df[col])
    return df


def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """Load, clean, encode, and save the dataset."""
    df = load_dataset(input_path)
    logger.info("Dataset contains %d records and %d features before processing.",
                len(df), len(df.columns))
    df.drop(columns=identify_columns_to_drop(df), inplace=True)
    df = encode_categorical_columns(df)
    df.to_csv(output_path, index=False)
    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df


# -------------------- NEW: model + metrics ---------------------------------- #
def train_and_evaluate(df: pd.DataFrame, metrics_path: str = "metrics.json") -> Dict[str, float]:
    """
    Train a logistic regression model using 5-fold CV and save metrics.
    Assumes target column is 'cons_general'.
    """
    if "cons_general" not in df.columns:
        logger.warning("Target column 'cons_general' not found; skipping metrics.")
        return {}

    y = df.pop("cons_general").to_numpy()
    y = np.where(y < 4, 0, 1)  # binarize
    X = df.to_numpy()

    # scale & impute
    X = StandardScaler().fit_transform(X)
    X = SimpleImputer(strategy="mean").fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    y_pred = cross_val_predict(clf, X, y, cv=5)

    acc = np.mean(y_pred == y)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    metrics = {"accuracy": acc, "specificity": specificity, "sensitivity": sensitivity}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s: %s", metrics_path, metrics)
    return metrics


def plot_summary(df: pd.DataFrame, project_name: str, output_path: str = "summary_plot.png") -> None:
    """Plot record counts by region with the project title."""
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
# Entry point
# --------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: preprocess, train, save metrics, and plot."""
    df = preprocess_dataset(
        input_path=cfg.experiment.dataset_file_path,
        output_path=cfg.experiment.dataset_file_preprocessed
    )
    # save metrics.json
    train_and_evaluate(df.copy(), metrics_path="output_metrics.json")
    # create a pretty plot
    plot_summary(df, project_name=cfg.experiment.name)


if __name__ == "__main__":
    main()
