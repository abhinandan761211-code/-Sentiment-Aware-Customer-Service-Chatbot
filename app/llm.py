"""Llama 3 client.

Talks to a locally running Ollama server over its HTTP API. Kept deliberately
small so the LLM can be swapped for another backend (llama.cpp, vLLM, an OpenAI
compatible endpoint) without touching the rest of the application.
"""

from __future__ import annotations

from typing import AsyncIterator, Dict, List, Optional

import httpx

from .config import settings
from .prompts import build_system_prompt, build_user_prompt
from .schemas import SentimentResult


class LlamaError(RuntimeError):
    """Raised when the Llama 3 backend cannot answer."""


class Llama3Client:
    """Thin async wrapper around Ollama's ``/api/chat`` endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.request_timeout

    # ── internals ────────────────────────────────────────────────────
    def _messages(
        self,
        message: str,
        sentiment: SentimentResult,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": build_system_prompt(sentiment)}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": build_user_prompt(message, sentiment)})
        return messages

    # ── health ───────────────────────────────────────────────────────
    async def health(self) -> str:
        """Return a short status string: 'ok', 'model-missing', 'unreachable'."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                tags = resp.json().get("models", [])
                names = {m.get("name", "").split(":")[0] for m in tags}
                if self.model.split(":")[0] in names:
                    return "ok"
                return "model-missing"
        except Exception:  # noqa: BLE001
            return "unreachable"

    # ── generation ───────────────────────────────────────────────────
    async def chat(
        self,
        message: str,
        sentiment: SentimentResult,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.6,
    ) -> str:
        """Generate one sentiment-appropriate reply."""
        payload = {
            "model": self.model,
            "messages": self._messages(message, sentiment, history),
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 300,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LlamaError(
                f"Could not reach Llama 3 at {self.base_url} ({exc}). "
                f"Start it with `ollama serve` and pull the model with "
                f"`ollama pull {self.model}`."
            ) from exc

        content = (data.get("message") or {}).get("content", "").strip()
        if not content:
            raise LlamaError("Llama 3 returned an empty response.")
        return content

    async def stream(
        self,
        message: str,
        sentiment: SentimentResult,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.6,
    ) -> AsyncIterator[str]:
        """Yield the reply token by token (used by the streaming endpoint)."""
        import json

        payload = {
            "model": self.model,
            "messages": self._messages(message, sentiment, history),
            "stream": True,
            "options": {"temperature": temperature, "top_p": 0.9, "num_predict": 300},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        piece = (chunk.get("message") or {}).get("content", "")
                        if piece:
                            yield piece
                        if chunk.get("done"):
                            break
        except httpx.HTTPError as exc:
            raise LlamaError(f"Streaming from Llama 3 failed: {exc}") from exc


# Module-level singleton — the client is stateless.
llama = Llama3Client()
