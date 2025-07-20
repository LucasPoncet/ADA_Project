import polars as pl
from pathlib import Path
import torch
import scipy.sparse as sp
import joblib

def load_split(split_name: str, processed_dir="data/processed", text_col="title", label_col="category_number"):
    path = Path(processed_dir) / f"{split_name}.parquet"
    df = pl.read_parquet(path)
    texts = df[text_col].to_list()
    labels = df[label_col].to_list()
    return texts, labels


def save_matrix(mat, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(mat, torch.Tensor):
        torch.save(mat, out_path)
    elif sp.issparse(mat):
        joblib.dump(mat, out_path)
    else:
        raise ValueError("Unknown matrix type")