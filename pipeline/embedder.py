"""
It will calculate the embeddings, and save them on disk. It also saves IDs and metadata.
"""
import json
import os

from pathlib import Path
import numpy as np
import pandas as pd

from datetime import datetime
import time
from sentence_transformers import SentenceTransformer

from .config import (
    ABSTRACTS_FILE,
    MODEL_NAME,
    EMBEDDINGS_FILE,
    IDS_FILE,
    METADATA_FILE,
)

from .data_loader import  load_abstracts

import mlflow

# the mlflow does not go into the docker. So, here, we define a boolean variable to disable the mlflow for the dockers, but to keep it for other pipelines.
DISABLE_MLFLOW = os.environ.get("DISABLE_MLFLOW", "0") == "1"


def calculate_embeddings():
    """
    parameters:
        path: path to the jsonl file
    Performs embedding calculation and saves it on disk.
    Saves IDs, and meta datas as well.
    """
    print("Loading abstracts...")

    df = load_abstracts()

    ids = df["id"].tolist()
    abstracts = df["abstract"].tolist()

    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # Recording time:
    start_time = datetime.now()
    print(f"Calculating embeddings...")
    embeddings = model.encode(abstracts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    duration = datetime.now() - start_time


    print(f"Savings embeddings to {EMBEDDINGS_FILE}")
    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"Saving array of IDs to {IDS_FILE}")

    ids_array = np.array(ids, dtype = int)
    np.save(IDS_FILE, ids_array)

    print(f"Saving metadata to {METADATA_FILE}")
    metadata = {
        "model_name": MODEL_NAME,
        "num_documents": str(len(ids)),
        "embeddings_dim": str(embeddings.shape[1]),
        "embeddings_filename": str(EMBEDDINGS_FILE),
        "ids_filename": str(IDS_FILE),
        "metadata_filename": str(METADATA_FILE),
        "created_at": datetime.now().isoformat(),
        "duration": duration.total_seconds(),
    }
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Embedding finished.")

    if not DISABLE_MLFLOW:
        print("MLFLOW logging is enabled...")

        mlflow.set_experiment("semantic-search-arxiv-abstracts")

        with mlflow.start_run(run_name="semantic-search-arxiv-abstracts"):
            mlflow.log_param("model_name", MODEL_NAME)
            mlflow.log_param("num_documents", len(ids))
            mlflow.log_param("embeddings_dim", int(embeddings.shape[1]))
            mlflow.log_param("duration", int(duration.total_seconds()))

            mlflow.log_artifact(str(ABSTRACTS_FILE))
            mlflow.log_artifact(str(EMBEDDINGS_FILE))
            mlflow.log_artifact(str(METADATA_FILE))
            mlflow.log_artifact(str(IDS_FILE))

        print("MLFLOW logging is done.")
    else:
        print("MLFLOW logging is disabled.")


if __name__ == "__main__":
    calculate_embeddings()

