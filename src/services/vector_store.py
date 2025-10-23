from __future__ import annotations

import json
import logging
from typing import Iterable, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from sqlalchemy import create_engine, text

from src.core.config import Settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Wrapper around PGVector providing helper utilities for ingestion and retrieval."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_path)
        self._vector_store = PGVector(
            connection_string=settings.connection_string,
            collection_name=settings.collection_name,
            embedding_function=self._embeddings,
            use_jsonb=True,
        )
        self._engine = create_engine(settings.connection_string)
        logger.info(
            "Vector store initialized | collection=%s | model_path=%s",
            settings.collection_name,
            settings.embedding_model_path,
        )

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings

    @property
    def store(self) -> PGVector:
        return self._vector_store

    def add_documents(self, documents: Iterable[Document]) -> None:
        docs = list(documents)
        if not docs:
            logger.warning("No documents provided for ingestion.")
            return

        self._vector_store.add_documents(docs)
        logger.info("Ingested %d documents into PGVector.", len(docs))

    def similarity_search(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[dict] = None,
    ) -> List[Document]:
        return self._vector_store.similarity_search(query, k=k, filter=metadata_filter)

    def load_all_documents(self) -> List[Document]:
        """Load all documents from the underlying PGVector collection."""

        query = text(
            """
            SELECT e.document, e.metadata
            FROM langchain_pg_embedding AS e
            INNER JOIN langchain_pg_collection AS c ON e.collection_id = c.id
            WHERE c.name = :collection
            """
        )
        with self._engine.connect() as connection:
            rows = connection.execute(query, {"collection": self._settings.collection_name})
            documents = []
            for row in rows:
                metadata = row.metadata
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {"source": metadata}
                documents.append(Document(page_content=row.document, metadata=metadata or {}))

        logger.info("Loaded %d documents from PGVector for local indexing.", len(documents))
        return documents


__all__ = ["VectorStoreService"]
