import os
from chromadb import Settings


CHROMA_SETTINGS = Settings(
    chroma_db_impl = 'duckdb-parquet',
    persist_directory = "db",
    anoymized_telemetry = False
)