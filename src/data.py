import polars as pl

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pl.read_parquet('hf://datasets/TimSchopf/arxiv_categories/' + splits['train'])
df2 = pl.read_parquet('hf://datasets/TimSchopf/arxiv_categories/' + splits['validation'])
df3 = pl.read_parquet('hf://datasets/TimSchopf/arxiv_categories/' + splits['test'])

df.write_parquet('data/raw/train.parquet')
df2.write_parquet('data/raw/validation.parquet')
df3.write_parquet('data/raw/test.parquet')