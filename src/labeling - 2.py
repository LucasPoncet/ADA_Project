import polars as pl
from pathlib import Path


GROUP_MAP = {
    "cs" : "Computer Science",
    "math" : "Mathematics",
    "physics" : "Physics",
    "stat" : "Statistics",
    "q-bio" : "Quantitative Biology",
    "q-fin" : "Quantitative Finance",
    "econ" : "Economics",
    "eess" : "Electrical Engineering",
}
GROUP_NUMBER_MAP = {
    "cs" : 0,
    "math" : 1,
    "physics" : 2,
    "stat" : 3,
    "q-bio" : 4,
    "q-fin" : 5,
    "econ" : 6,
    "eess" : 7,
}

def map_category(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("categories")
          .list.first()
          .str.replace(" Archive", "", literal=True)
          .str.split(by="->")
          .list.get(0)
          .map_elements(
              lambda s: next(
                  (k for k, v in GROUP_MAP.items() if s.startswith(v)),
                  "Other"
              ),
              return_dtype=pl.Utf8                  
          )
          .alias("category")
    )

def map_category_number(category: str) -> int:
    return GROUP_NUMBER_MAP.get(category, -1)

# Load the data
DATA= Path("data/raw/")
train_pl = pl.read_parquet(DATA / "train.parquet")
val_pl = pl.read_parquet(DATA / "validation.parquet")
test_pl = pl.read_parquet(DATA / "test.parquet")

# Apply the mapping to the 'categories' column
train_pl = map_category(train_pl)
val_pl = map_category(val_pl)
test_pl = map_category(test_pl)

# Put Categories in number for the model
train_pl = train_pl.with_columns(
    pl.col("category").map_elements(map_category_number, return_dtype=pl.Int64).alias("category_number")
)
val_pl = val_pl.with_columns(
    pl.col("category").map_elements(map_category_number, return_dtype=pl.Int64).alias("category_number")
)
test_pl = test_pl.with_columns(
    pl.col("category").map_elements(map_category_number, return_dtype=pl.Int64).alias("category_number")
)

# Save the processed data
PROCESSED_DATA = Path("data/processed/")
if not PROCESSED_DATA.exists():
    PROCESSED_DATA.mkdir(parents=True)

train_pl.write_parquet(PROCESSED_DATA / "train_processed.parquet")
val_pl.write_parquet(PROCESSED_DATA / "validation_processed.parquet")
test_pl.write_parquet(PROCESSED_DATA / "test_processed.parquet")