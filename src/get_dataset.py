import os
import requests
from tqdm import tqdm
import logging
import hydra

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Create a logger
logger = logging.getLogger(__name__) 

# Load configuration using Hydra
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    DATASET_URL = cfg.experiment.dataset_url
    DATASET_FOLDER = cfg.experiment.dataset_folder
    DATASET_FILE_PATH = cfg.experiment.dataset_file_path

    # Create data directory if it doesn't exist
    logger.info(f"Creating dataset folder at {DATASET_FOLDER} if it doesn't exist.")
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    # Download the dataset
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    # Write the dataset to a file with a progress bar
    logger.info(f"Downloading dataset from {DATASET_URL} to {DATASET_FILE_PATH}.")
    with open(DATASET_FILE_PATH, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192), unit='B', unit_scale=True):
            f.write(chunk)

    # Confirm download completion
    logger.info(f"Dataset downloaded and saved to {DATASET_FILE_PATH}.")

if __name__ == "__main__":
    main()