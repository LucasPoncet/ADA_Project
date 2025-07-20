# src/utils/embedding_utils.py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# TF-IDF
def tfidf_encode(
    texts: list[str],
    fit: bool,
    vec_path,                      # pathlib.Path ou str
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int,
    sublinear_tf: bool,
    stop_words: str = "english",
):

    if fit:
        vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            stop_words=stop_words,
        )
        X = vec.fit_transform(texts)
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        vec_path = str(vec_path)
        vec_path = vec_path if vec_path.endswith(".pkl") else vec_path + ".pkl"
        joblib.dump(vec, vec_path)
    else:
        vec = joblib.load(vec_path)
        X = vec.transform(texts)
    return X


# BERT / SciBERT
def bert_encode(
    texts: list[str],
    model_name: str,
    device: str | torch.device,
    batch_size: int,
    max_length: int,
):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)

    def batches(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    cls_vectors = []
    mdl.eval()
    with torch.no_grad():
        for batch in tqdm(batches(texts, batch_size),
                          total=(len(texts) + batch_size - 1) // batch_size,
                          desc="BERT encode"):
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = mdl(**enc).last_hidden_state[:, 0, :]   # vecteur CLS
            cls_vectors.append(out.cpu())
    return torch.cat(cls_vectors, dim=0)                 # (N, hidden_size)
