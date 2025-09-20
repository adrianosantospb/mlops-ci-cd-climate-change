import os
import requests
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Create a logger
logger = logging.getLogger(__name__) 

# Dataset URL and local file paths
DATASET_URL = 'https://www.research-collection.ethz.ch/bitstreams/42a2f58b-d925-4352-908f-91db854466a1/download'
DATASET_FILE_NAME = "dataset.csv"
DATASET_FOLDER = "data"
DATASET_FILE_PATH = os.path.join(DATASET_FOLDER, DATASET_FILE_NAME)

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