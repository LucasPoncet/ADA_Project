import pathlib
from typing import TypedDict
import joblib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import torch
import scipy.sparse as sp
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import pandas as pd
import xgboost as xgb
import seaborn as sns
class HP(TypedDict):
    method      : str
    emb_path    : str
    model_path  : str
    out_dir     : str
    label_names : list[str]
    label_col   : str
    parquet_path: str

hp: HP   = HP(
    method      = "xgb",                              # "tfidf" or "bert" or "xgb"
    emb_path    = "data/embeddings/test_processed_bert.pkl",
    model_path  = "models/classifiers/xgb_cls.json",
    out_dir     = "experiments/eval_xgb_bert",
    label_names = [
        "astro-ph",
        "cond-mat",
        "cs",
        "econ",
        "math",
        "physics",
        "q-bio",
    ],
    label_col   = "category_number",                    # column in parquet
    parquet_path= "data/processed/test_processed.parquet",
)


out = pathlib.Path(hp["out_dir"]) 
out.mkdir(parents=True, exist_ok=True)

# load X 
if hp["method"] == "bert" or hp["method"] == "xgb":
    X_test = torch.load(hp["emb_path"]).float()


else:
    X_test = joblib.load(hp["emb_path"])
    if sp.issparse(X_test):
        X_test = X_test.astype(np.float32)

#  load y
y_test = (
    pl.read_parquet(hp["parquet_path"], columns=[hp["label_col"]])[hp["label_col"]]
    .to_numpy()
)

# load model & predict
if hp["method"] == "bert":
    from ClassesML.BertClassifier import BertClassifier

    num_classes = int(y_test.max()) + 1
    model = BertClassifier(X_test.shape[1], hidden=512, n_classes=num_classes)
    model.load_state_dict(torch.load(hp["model_path"], map_location="cpu"))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1).numpy()

elif hp["method"] == "xgb":
    dtest = xgb.DMatrix(X_test)
    model = xgb.Booster()
    model.load_model(hp["model_path"])
    y_pred = np.argmax(model.predict(dtest), axis=1)
    
else:
    clf = joblib.load(hp["model_path"])
    y_pred = clf.predict(X_test)

# metrics
acc   = accuracy_score(y_test, y_pred)
m_f1  = f1_score(y_test, y_pred, average="macro")
print(f"TEST accuracy = {acc:.4f}   macro-F1 = {m_f1:.4f}\n")

report_dict = classification_report(
    y_test, y_pred, target_names=hp["label_names"], digits=3, output_dict=True
)
print(classification_report(y_test, y_pred, target_names=hp["label_names"], digits=3))

# save CSV

pd.DataFrame(report_dict).T.to_csv(out / "classification_report.csv", index=True)

# confusion matrix

cm = confusion_matrix(y_test, y_pred, normalize="true")

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="plasma",       
    xticklabels=hp["label_names"],
    yticklabels=hp["label_names"],
    cbar_kws={"label": "Recall"},
    ax=ax,
)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Normalized Confusion Matrix")

plt.tight_layout()
fig_path = out / "confusion_matrix_plasma.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print("CSV & confusion matrix saved to", fig_path)
