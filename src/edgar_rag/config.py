"""Runtime settings, loaded from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # The SEC asks every client to identify itself with a name and email.
    sec_user_agent: str = field(
        default_factory=lambda: os.getenv("SEC_USER_AGENT", "edgar-rag research contact@example.com")
    )
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("EDGAR_RAG_DATA", "data")))

    # Any OpenAI-compatible chat endpoint works (Ollama by default).
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.1:8b"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    child_chunk_words: int = 220
    child_overlap_words: int = 40
    parent_chunk_words: int = 1200

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


settings = Settings()
