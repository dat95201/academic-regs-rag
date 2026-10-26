"""Runtime settings, read from environment variables / .env (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str = "qwen3:4b"
    embed_model: str = "bge-m3"
    ollama_url: str = "http://ollama:11434"
    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "quy_che"

    chunk_max_tokens: int = 800
    # Paragraphs repeated at the start of the next piece when a section must be cut.
    chunk_overlap_tokens: int = 100

    top_k_retrieve: int = 20
    top_k_final: int = 5
    hybrid_search: bool = True
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    min_relevance_score: float = 0.3
    llm_temperature: float = 0.2
    max_context_tokens: int = 4000

    processed_dir: str = "data/processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
