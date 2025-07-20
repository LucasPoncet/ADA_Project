import os
import pathlib
import torch

from utils.data_utils      import load_split, save_matrix
from utils.embedding_utils import tfidf_encode, bert_encode

# General settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = pathlib.Path(os.getcwd())

# Hyper-parameters
hyperparameter: dict = dict(
    # Choice of method and split
    method          = "bert",                   # "tfidf" or "bert"
    split           = "test_processed",                   # "train_processed" / "validation_processed" / "test_processed"

    # TF-IDF
    max_features    = 50_000,
    ngram_range     = (1, 2),
    min_df          = 5,
    sublinear_tf    = True,

    # BERT / SciBERT
    model_name      = "allenai/scibert_scivocab_uncased",
    batch_size      = 16,
    max_length      = 256,
    device          = device,                   # "cpu" or "cuda"

    # Paths
    out_dir         = project_root / "data" / "embeddings",
    vec_path        = project_root / "models" / "vectorizers" / "tfidf.pkl",

    # Parquet columns
    text_col        = "title",
    label_col       = "category_number",
)

# Loading of the split
texts, _ = load_split(
    split_name        = hyperparameter["split"],
    processed_dir= "data/processed",
    text_col     = hyperparameter["text_col"],
    label_col    = hyperparameter["label_col"],
)
fit = hyperparameter["split"] == "train_processed"

# Encoder TF-IDF or BERT
if hyperparameter["method"] == "tfidf":
    X = tfidf_encode(
        texts           = texts,
        fit             = fit,
        vec_path        = hyperparameter["vec_path"],
        max_features    = hyperparameter["max_features"],
        ngram_range     = hyperparameter["ngram_range"],
        min_df          = hyperparameter["min_df"],
        sublinear_tf    = hyperparameter["sublinear_tf"],
        stop_words      = "english",
    )
else:  # BERT
    X = bert_encode(
        texts       = texts,
        model_name  = hyperparameter["model_name"],
        device      = hyperparameter["device"],
        batch_size  = hyperparameter["batch_size"],
        max_length  = hyperparameter["max_length"],
    )

# Save the matrix
out_path = pathlib.Path(hyperparameter["out_dir"]) / \
           f"{hyperparameter['split']}_{hyperparameter['method']}.pkl"
save_matrix(X, out_path)

# Log minimal
print(f"✓ {hyperparameter['split']} {hyperparameter['method']}  →  {out_path}")
print(f"  shape = {tuple(X.shape)}   dtype = {X.dtype  if hasattr(X,'dtype') else 'sparse'}")# type: ignore[attr-defined]
