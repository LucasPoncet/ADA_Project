import polars as pl
from pathlib import Path

DATA_DIR   = Path("data/raw/")         
TRAIN_PQ   = DATA_DIR / "train.parquet"
VAL_PQ     = DATA_DIR / "validation.parquet"
TEST_PQ    = DATA_DIR / "test.parquet"

train_pl = pl.read_parquet(TRAIN_PQ)
val_pl   = pl.read_parquet(VAL_PQ)
test_pl  = pl.read_parquet(TEST_PQ)


for name, df in [("train", train_pl), ("val", val_pl), ("test", test_pl)]:
    print(f"{name:5s} → rows={df.height:,}  cols={df.width}")
    print(df.schema, "\n")

def null_report(df: pl.DataFrame):
    return (df
            .null_count().with_columns(pl.col("*").cast(pl.Float32)/df.height))


print(null_report(train_pl))
print(null_report(val_pl))
print(null_report(test_pl))

train_pl.group_by("categories").len().sort("len", descending=True)

(train_pl
 .group_by("categories")
 .agg(pl.count().alias("count"))
 .sort("count", descending=True)
 .head(20)
 .select(["count", "categories"])
 .write_parquet("data/samples/snapshot_sample.parquet"))