# Train XGBoost (CPU-only) on BERT CLS embeddings.

import pathlib
import torch
import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from typing import TypedDict

class HP (TypedDict):
    train_emb: str
    val_emb: str
    model_out: str
    parquet_col: str
    train_parquet: str
    val_parquet: str
    params: dict
    rounds: int
    early_stop_round: int

hp: HP = HP(
    train_emb = "data/embeddings/train_processed_bert.pkl",
    val_emb   = "data/embeddings/validation_processed_bert.pkl",
    model_out = "models/classifiers/xgb_cls3.json",
    parquet_col = "category_number",
    train_parquet = "data/processed/train_processed.parquet",
    val_parquet   = "data/processed/validation_processed.parquet",
    params = dict(                 
        max_depth        = 8,
        eta              = 0.1,   # learning-rate
        subsample        = 0.8,
        colsample_bytree = 0.6,
        min_child_weight = 5,
        objective        = "multi:softprob",
        num_class        = 8,
        eval_metric      = "mlogloss",
        nthread          = 8,      # adapt to your CPU core count ( On my macbook, 8 cores )
        seed             = 100,
    ),
    rounds          = 400,
    early_stop_round= 25,
)

# data loaders 
def load_emb(path):           
    return torch.load(path).cpu().numpy().astype(np.float32)

def load_y(parquet_path, col):
    return pl.read_parquet(parquet_path, columns=[col])[col].to_numpy()

X_tr = load_emb(hp["train_emb"])
X_va = load_emb(hp["val_emb"])
y_tr = load_y(hp["train_parquet"], hp["parquet_col"])
y_va = load_y(hp["val_parquet"],   hp["parquet_col"])

freq = np.bincount(y_tr, minlength=8) / len(y_tr)
weights = 1.0 / (freq + 1e-6)                
w_train = weights[y_tr]                 
w_val   = weights[y_va]          

dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_train)
dval   = xgb.DMatrix(X_va, label=y_va, weight=w_val)


#  training 
model = xgb.train(
    params               = hp["params"],
    dtrain               = dtrain,
    num_boost_round      = hp["rounds"],
    evals                = [(dval, "val")],
    early_stopping_rounds= hp["early_stop_round"],
    verbose_eval         = 50,        # prints every 50 rounds
)

# validation metrics
val_pred = np.argmax(model.predict(dval), axis=1)
acc      = accuracy_score(y_va, val_pred)
macro_f1 = f1_score(y_va, val_pred, average="macro")
print(f"\nVAL  accuracy = {acc:.4f}   macro-F1 = {macro_f1:.4f}")

# save model

out_path = pathlib.Path(hp["model_out"])
out_path.parent.mkdir(parents=True, exist_ok=True)
model.save_model(out_path)
print("XGBoost model saved to", out_path)
