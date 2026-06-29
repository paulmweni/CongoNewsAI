"""Ingestion entry point for the News AI pipeline.

Runs the full ingestion flow: parse the news CSV, chunk the articles, embed
the chunks, and persist them to the Chroma vector store. Run this once (or
whenever the dataset changes) before querying via ``cli.py`` or ``app.py``.
"""

import pandas as pd
from pathlib import Path

from src.parser import NewsParser
from src.chunker import NewsChanker
from src.embed import NewsEmbedder

# ---- Settings ----
CSV_PATH = "data/actu_cd.csv"
OUTPUT_CSV = "data/parsed_articles.csv"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "news"
# ------------------

parser = NewsParser(CSV_PATH)
df = parser.parse()
chunker = NewsChanker()
documents = chunker.chunk_dataframe(df)
embedder = NewsEmbedder(PERSIST_DIR, COLLECTION_NAME)
embedder.add_documents(documents)