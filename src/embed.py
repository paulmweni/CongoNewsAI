"""Embedding and vector-store persistence for the News AI pipeline.

Exposes :class:`NewsEmbedder`, which embeds chunked Documents with a local
HuggingFace model and stores them in a persistent Chroma vector database.
"""

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
from typing import List, Optional


class NewsEmbedder:
    """Embeds chunked news Documents into a persistent Chroma vector store."""

    def __init__(
            self,
            persist_directory: str = "chroma_db",
            collection_name: str = "news",
            model_name: str = "all-MiniLM-L6-v2",
            batch_size: int = 5000
    ):
        """Initialise the embedding model and Chroma vector store.

        Args:
            persist_directory: Directory where the Chroma database is stored.
            collection_name: Name of the Chroma collection.
            model_name: HuggingFace sentence-transformers model used for embeddings.
            batch_size: Number of documents to embed per batch.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.batch_size = batch_size

        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
    def add_documents(self, documents: List[Document]) -> None:
        """Embed and store the given documents in batches.

        Args:
            documents: The chunked Documents to embed and persist.
        """
        print(f"[✓] Starting embedding: {len(documents)} documents to process...")

        for i in tqdm(range(0, len(documents), self.batch_size), desc="Embedding batches"):
            batch = documents[i:i + self.batch_size]
            self.vectorstore.add_documents(batch)

        print(f"[✓] Embedding completed and saved to {self.persist_directory}")

    def get_vectorstore(self) -> Chroma:
        """
        Returns the internal Chroma vectorstore.
        Useful for querying later.

        Returns:
            Chroma: The Chroma vector store instance.
        """
        return self.vectorstore

