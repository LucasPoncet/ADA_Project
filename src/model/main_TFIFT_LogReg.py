import joblib
import pathlib
import numpy as np
import scipy.sparse as sp
from itertools import product
from typing import TypedDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import polars as pl

class HPTFIDF(TypedDict):
    train_emb: str
    val_emb  : str
    test_emb : str
    model_out: str
    max_iter : int
    C        : float

hp: HPTFIDF = {
    "train_emb": "data/embeddings/train_processed_tfidf.pkl",
    "val_emb"  : "data/embeddings/validation_processed_tfidf.pkl",
    "test_emb" : "data/embeddings/test_processed_tfidf.pkl",
    "model_out": "models/classifiers/logreg_tfidf.pkl",
    "max_iter" : 400,
    "C"       : 2.0,
}

pathlib.Path(hp["model_out"]).parent.mkdir(parents=True, exist_ok=True)
X_train = joblib.load(hp["train_emb"])
X_val   = joblib.load(hp["val_emb"])

def load_labels(split: str, col: str = "category_number"):
    df = pl.read_parquet(f"data/processed/{split}.parquet", columns=[col])
    return df[col].to_numpy()

y_train = load_labels("train_processed")   # adapte au nom réel de ton split
y_val   = load_labels("validation_processed")

# option : cast in float32 for RAM
if sp.issparse(X_train):
    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)

# grid definition
C_grid            = [1.0, 2.0, 4.0]
class_weight_grid = [None, "balanced"]

best_f1   = -1.0
best_acc  = -1.0
best_clf  = None
best_clf_acc = None
best_desc = ""

for C, cw in product(C_grid, class_weight_grid):
    print(f"\n─> C={C}  class_weight={cw}")

    clf = LogisticRegression(
        max_iter     = hp["max_iter"],
        C            = C,
        solver       = "saga",
        multi_class  = "multinomial",
        class_weight = cw,
        n_jobs       = -1,
        verbose      = 0,
    )
    clf.fit(X_train, y_train)
    val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"   val macro-F1 = {val_f1:.4f}   val Acc = {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        best_clf_acc = clf
    if val_f1 > best_f1:
        best_f1, best_clf = val_f1, clf
        best_desc = f"C={C}, cw={cw or 'none'}"

# save once 
joblib.dump(best_clf, hp["model_out"])
print("\n████  RÉSUMÉ  ████")
print(f"Meilleur val macro-F1 = {best_f1:.4f}  ({best_desc})")
print(f"Modèle enregistré     = {hp['model_out']}")

# Save the best model with accuracy
if best_clf_acc is not None:
    acc_out_path = hp["model_out"].replace(".pkl", "_acc.pkl")
    joblib.dump(best_clf_acc, acc_out_path)
    print(f"Modèle avec meilleure accuracy enregistré = {acc_out_path}")


