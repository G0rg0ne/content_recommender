import polars as pl
from FlagEmbedding import BGEM3FlagModel
import numpy as np
import os 

DATA_DIR  = "./data/ml-25m"  
OUTPUT_DIR = "./data/movie_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

movies = pl.read_csv(f"{DATA_DIR}/movies.csv",
                     schema_overrides={
                         "movieId": pl.Int32,
                         "title":   pl.Utf8,
                         "genres":  pl.Utf8
                     })

#Dense (1024) and sparse (lexical) representations return_dense=True, return_sparse=True
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
soup_strings = (
    movies.select(
        pl.col("title") + " " + pl.col("genres").str.replace_all(r"\|", " ")
    )
    .to_series()
    .to_list()
)
embeddings = model.encode(soup_strings)

dense = embeddings["dense_vecs"]
assert len(dense) == len(movies)

out = movies.with_columns(
    embedding=pl.Series(
        name="embedding",
        values=dense.astype(np.float32, copy=False),
        dtype=pl.List(pl.Float32),  # list[f32] per row; fixed length 1024 per row
    )
)

out.write_parquet(
    f"{OUTPUT_DIR}/movies_bge_m3_dense.parquet",
    compression="zstd",
)