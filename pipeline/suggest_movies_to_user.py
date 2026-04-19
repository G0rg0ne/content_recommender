import polars as pl
import numpy as np
import os 


EMBEDDING_DIR = "./data/movie_embeddings"
DATA_DIR = "./data/ml-25m"

ratings = pl.read_csv(f"{DATA_DIR}/ratings.csv",
                      schema_overrides={
                          "userId": pl.Int32,
                          "movieId": pl.Int32,
                          "rating": pl.Float32,
                          "timestamp": pl.Int64
                      })
