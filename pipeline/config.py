"""
A simple config file to set up directories, filenames, and model information
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ABSTRACTS_FILE = DATA_DIR / "arxiv_abstracts.jsonl"

MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDINGS_FILE = MODELS_DIR / "embeddings.npy"
IDS_FILE = MODELS_DIR / "ids.npy"
METADATA_FILE = MODELS_DIR / "metadata.json"
