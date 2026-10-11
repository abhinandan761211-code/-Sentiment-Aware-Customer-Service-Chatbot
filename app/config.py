"""Environment-driven configuration for the sentiment-aware chatbot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):  # type: ignore
        return False

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3"))

    # auto | transformer | lexicon
    sentiment_backend: str = field(default_factory=lambda: _env("SENTIMENT_BACKEND", "auto"))
    sentiment_model: str = field(
        default_factory=lambda: _env(
            "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    )

    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8000")))

    # How many prior turns to replay to Llama 3 as conversation memory.
    history_turns: int = field(default_factory=lambda: int(_env("HISTORY_TURNS", "6")))

    request_timeout: float = field(default_factory=lambda: float(_env("REQUEST_TIMEOUT", "60")))


settings = Settings()

STATIC_DIR = BASE_DIR / "static"
EVAL_DIR = BASE_DIR / "evaluation"
