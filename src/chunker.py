"""Text chunking for the News AI pipeline.

Exposes :class:`NewsChanker`, which turns a DataFrame of news articles into a
list of LangChain :class:`~langchain_core.documents.Document` objects, split
into overlapping chunks suitable for embedding.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pandas as pd


class NewsChanker:
    """Splits news articles into overlapping text chunks with metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        """Configure the recursive character text splitter.

        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between adjacent chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_dataframe(self, df: pd.DataFrame, text_column: str = "article_texts") -> list[Document]:
        """Split each article in the DataFrame into chunked Documents.

        Rows whose text column is missing or blank are skipped. Each resulting
        chunk carries the article's title, date, and URL as metadata.

        Args:
            df: DataFrame of news articles.
            text_column: Name of the column holding the article body text.

        Returns:
            A list of chunked LangChain Documents ready for embedding.
        """
        documents = []

        for _, row in df.iterrows():
            text = row[text_column]
            if not isinstance(text, str) or not text.strip():
                continue

            metadata = {
                "title": row.get("title"),
                "date": row.get("date"),
                "url": row.get("link"),
            }

            chunks = self.splitter.create_documents([text], metadatas=[metadata])
            documents.extend(chunks)

        return documents
        


