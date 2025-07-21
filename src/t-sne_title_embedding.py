import pathlib
import torch
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


EMB_PATH   = "data/embeddings/validation_processed_bert.pkl"   # any split
PARQUET    = "data/processed/validation_processed.parquet"
LABEL_COL  = "category_number"
LABEL_NAMES= ["astro-ph","cond-mat","cs","econ","math","physics","q-bio","q-fin"]
OUT_DIR    = pathlib.Path("experiments/figures")
N_POINTS   = 10_000            # subsample for speed / clarity
RANDOM_SEED= 100


OUT_DIR.mkdir(parents=True, exist_ok=True)

#load & subsample
rng      = np.random.default_rng(RANDOM_SEED)
X_full   = torch.load(EMB_PATH).cpu().numpy()
y_full   = pl.read_parquet(PARQUET, columns=[LABEL_COL])[LABEL_COL].to_numpy()

idx      = rng.choice(len(X_full), size=min(N_POINTS, len(X_full)), replace=False)
X, y     = X_full[idx], y_full[idx]

# t-SNE
print("Running t-SNE … (this can take ~1 min)")
X2 = TSNE(n_components=2, init="pca", perplexity=30, random_state=RANDOM_SEED).fit_transform(X)

# plot
fig, ax = plt.subplots(figsize=(7, 6))
cmap = plt.get_cmap("tab10")
for k, name in enumerate(LABEL_NAMES):
    pts = X2[y == k]
    ax.scatter(pts[:,0], pts[:,1], s=8, color=cmap(k), label=name, alpha=0.7)

ax.set_xticks([]) 
ax.set_yticks([])
ax.legend(markerscale=2, fontsize=8, loc="upper right")
ax.set_title("t-SNE of SciBERT CLS embeddings (validation split)")

out_file = OUT_DIR / "tsne_cls_validation.png"
plt.tight_layout() 
plt.savefig(out_file, dpi=300)
print("figure saved →", out_file)
