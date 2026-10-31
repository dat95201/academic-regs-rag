"""Runtime settings, read from environment variables / .env (see .env.example)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- models ---
    # "ollama" = local model below; "openai" = cloud chat model (OPENAI_MODEL, needs OPENAI_API_KEY).
    # Embeddings always stay on Ollama, so switching provider needs no re-ingest.
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:4b"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    embed_model: str = "bge-m3"
    ollama_url: str = "http://ollama:11434"
    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "quy_che"

    # --- chunking ---
    chunk_max_tokens: int = 800
    # Paragraphs repeated at the start of the next piece when a section must be cut.
    chunk_overlap_tokens: int = 100

    # --- retrieval ---
    top_k_retrieve: int = 20
    top_k_final: int = 5
    hybrid_search: bool = True
    # Weights of the two branches in the rank fusion. Dense finds paraphrases, sparse finds
    # the exact wording; dense is trusted slightly more because the BM25 terms are hashed.
    dense_weight: float = 1.0
    sparse_weight: float = 0.8
    # Reranking the candidates before they reach the model. "heuristic" is free and local
    # (term coverage + title match + article match + diversity); "llm" spends one extra call
    # on the chat model; "off" keeps the fusion order. No cross-encoder, so no torch.
    rerank_enabled: bool = True
    rerank_mode: str = "heuristic"  # off | heuristic | llm
    rerank_candidates: int = 20
    rerank_model: str = ""  # unused; kept so an old .env does not fail to load
    # Nothing below this cosine counts as an answer: better to say so than to invent one.
    min_relevance_score: float = 0.3
    # ...and nothing much worse than the best hit does either. bge-m3 cosines sit in a narrow
    # band (a random paragraph still scores ~0.45), so an absolute floor alone lets four
    # padding chunks ride along behind one good one. This keeps a chunk only while it stays
    # within RELATIVE_CUTOFF of the best chunk found, which is what makes the number of
    # sources vary with the question instead of always being TOP_K_FINAL.
    relative_cutoff: float = 0.85
    # "GPA", "rớt môn", "hoc phan" -> the wording the regulation itself uses.
    glossary_expansion: bool = True
    glossary_for_dense: bool = True
    # "Điều 15 nói gì?" -> fetch Điều 15 directly instead of hoping the vectors agree.
    article_lookup: bool = True
    # Widen each kept chunk with its neighbours in the same article (15.3 answers with 15.4).
    context_expansion: bool = True
    context_neighbours: int = 2
    # Repeated questions skip the embedding call.
    query_cache_size: int = 256

    # --- generation ---
    llm_temperature: float = 0.2
    max_context_tokens: int = 4000
    # OpenAI-compatible endpoint; empty means Ollama's (<OLLAMA_URL>/v1). Point it at vLLM later.
    llm_base_url: str = ""
    llm_api_key: str = "ollama"
    # Qwen3 "thinking" is slow and not needed here; adds the /no_think switch and strips <think>.
    llm_disable_thinking: bool = True
    # Past user/assistant turns sent along with a follow-up question.
    history_turns: int = 3
    # A follow-up is rewritten into a standalone question before searching. That is an extra
    # LLM round trip, so it is skipped when the question already stands on its own.
    rewrite_follow_ups: bool = True

    # --- operations ---
    # Questions, latency and whether an answer was found, appended as JSON lines. This is what
    # tells you which questions the corpus cannot answer; see GET /metrics.
    log_queries: bool = True
    log_dir: str = "data/logs"

    @property
    def use_openai(self) -> bool:
        return self.llm_provider.strip().lower() == "openai"

    @property
    def llm_url(self) -> str:
        if self.use_openai:
            return self.openai_base_url.rstrip("/")
        return (self.llm_base_url or f"{self.ollama_url}/v1").rstrip("/")

    @property
    def chat_model(self) -> str:
        return self.openai_model if self.use_openai else self.llm_model

    @property
    def chat_api_key(self) -> str:
        return self.openai_api_key if self.use_openai else self.llm_api_key

    @property
    def send_no_think(self) -> bool:
        # "/no_think" is a Qwen3 switch; a cloud model would just see it as text.
        return self.llm_disable_thinking and not self.use_openai

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
