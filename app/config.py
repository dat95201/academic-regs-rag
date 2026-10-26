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
    # OpenAI-compatible endpoint; empty means Ollama's (<OLLAMA_URL>/v1). Point it at vLLM later.
    llm_base_url: str = ""
    llm_api_key: str = "ollama"
    # Qwen3 "thinking" is slow and not needed here; adds the /no_think switch and strips <think>.
    llm_disable_thinking: bool = True
    # Past user/assistant turns sent along with a follow-up question.
    history_turns: int = 3

    @property
    def llm_url(self) -> str:
        return (self.llm_base_url or f"{self.ollama_url}/v1").rstrip("/")

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    # Which file version is in Qdrant; lets ingest skip unchanged documents.
    ingest_manifest: str = "data/.ingest_manifest.json"

    embed_batch_size: int = 16
    # CPU-only machines can take minutes for the first call while the model loads.
    ollama_timeout_s: float = 600


@lru_cache
def get_settings() -> Settings:
    return Settings()
