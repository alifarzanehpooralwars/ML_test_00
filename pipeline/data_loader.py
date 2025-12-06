import pandas as pd
import numpy as np
import json
from pathlib import Path

from .config import ABSTRACTS_FILE

def load_abstracts(path: Path = ABSTRACTS_FILE, n_rows: int | None = None) -> pd.DataFrame:
    """
    A function to read ids and abstracts from the json file.
    Params:
        path: Path to the json file.
        n_rows: If provided, returns n_rows rows of the data.
    Returns:
        A pandas DataFrame containing the ids and abstracts.
    """

    # check path (file)
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found")

    df = pd.read_json(path, lines=True)

    if "id" not in df.columns or "abstract" not in df.columns:
        raise FileNotFoundError(f"File {path} not found")

    if n_rows:
        return df.iloc[:n_rows].copy()

    return df

if __name__ == "__main__":
    df = load_abstracts(n_rows = 5)
    print(df)
