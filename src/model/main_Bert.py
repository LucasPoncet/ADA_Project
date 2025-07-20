import pathlib
import torch
import polars as pl
from typing import TypedDict
from ClassesML.BertClassifier import BertClassifier
from ClassesML.TrainerClassifier       import TrainerClassifier


class HPBert(TypedDict):
    train_emb: str
    val_emb  : str
    model_out: str
    hidden   : int
    lr       : float
    epochs   : int
    patience : int


hp: HPBert = {
    "train_emb": "data/embeddings/train_processed_bert.pkl",
    "val_emb"  : "data/embeddings/validation_processed_bert.pkl",
    "model_out": "models/classifiers/bert_mlp.pt",
    "hidden"   : 512,
    "lr"       : 1e-3,
    "epochs"   : 50,
    "patience" : 5,
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pathlib.Path(hp["model_out"]).parent.mkdir(parents=True, exist_ok=True)

# load embeddings (torch tensors)
X_train = torch.load(hp["train_emb"]).to(device).float()
X_val   = torch.load(hp["val_emb"]).to(device).float()

def load_labels(split: str, col: str = "category_number"):
    df = pl.read_parquet(f"data/processed/{split}.parquet", columns=[col])
    return df[col].to_numpy()

y_train = load_labels("train_processed")
y_val   = load_labels("validation_processed")

X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
X_val,   y_val   = torch.tensor(X_val),   torch.tensor(y_val)

n_classes = len(torch.unique(y_train))+1

model = BertClassifier(in_dim=X_train.shape[1],
                       hidden=hp["hidden"],
                       n_classes=n_classes).to(device)

optim  = torch.optim.Adam(model.parameters(), lr=hp["lr"])
lossfn = torch.nn.CrossEntropyLoss()

trainer = TrainerClassifier(model, optim, lossfn, device, patience=hp["patience"])
train_hist, val_hist, best_f1 = trainer.run(X_train, y_train, X_val, y_val, epochs=hp["epochs"])

torch.save(model.state_dict(), hp["model_out"])
print(f"✓ saved {hp['model_out']}  | best val macro-F1 = {best_f1:.4f}")
