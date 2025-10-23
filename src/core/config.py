from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    connection_string: str = Field(
        default="postgresql+psycopg2://postgres@cbq2-svd-daldb1:1830/vector",
        alias="CONNECTION_STRING",
        description="SQLAlchemy connection string for the PGVector database.",
    )
    embedding_model_path: str = Field(
        default=r"D:\\EmbeddingModel\\instructor-xl",
        alias="EMBEDDING_MODEL_PATH",
        description="Absolute path to the local HuggingFace embedding model.",
    )
    ollama_base_url: str = Field(
        default="http://10.1.94.110:11434",
        alias="OLLAMA_BASE_URL",
        description="Base URL of the Ollama server.",
    )
    ollama_model_id: str = Field(
        default="llama3:70b",
        alias="OLLAMA_MODEL_ID",
        description="Model identifier to use for the Ollama chat model.",
    )
    collection_name: str = Field(
        default="deep_thinking_rag",
        description="Collection name used inside the PGVector database.",
    )
    chunk_size: int = Field(default=1000, description="Character count for each text chunk during ingestion.")
    chunk_overlap: int = Field(default=150, description="Number of overlapping characters between chunks.")
    top_k_retrieval: int = Field(default=10, description="Number of documents retrieved for recall stage.")
    top_n_rerank: int = Field(default=3, description="Number of top documents kept after reranking.")
    max_reasoning_iterations: int = Field(default=7, description="Maximum iterations for the reasoning loop.")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="SentenceTransformers cross-encoder model name used for reranking.",
    )
    enable_web_search: bool = Field(
        default=False,
        description="Whether to enable the Tavily web search integration. Requires TAVILY_API_KEY environment variable.",
    )
    tavily_max_results: int = Field(
        default=3,
        description="Number of Tavily search results to fetch when web search is enabled.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached instance of application settings."""

    return Settings()


__all__ = ["Settings", "get_settings"]
