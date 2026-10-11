"""FastAPI application: the sentiment-aware chatbot service.

Pipeline per request:

    user message → sentiment analysis → sentiment-aware prompt → Llama 3 → reply

The chat UI in ``static/`` is served from the same process.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict, deque
from typing import Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import STATIC_DIR, settings
from .llm import LlamaError, llama
from .prompts import SENTIMENT_POLICY
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ResetRequest,
    SentimentOnlyRequest,
    SentimentResult,
)
from .sentiment import SentimentEngine, response_is_appropriate

app = FastAPI(
    title="Sentiment-Aware Customer Service Chatbot",
    description=(
        "Detects POSITIVE / NEUTRAL / NEGATIVE sentiment in customer messages "
        "and has Llama 3 respond appropriately to each emotion."
    ),
    version="1.0.0",
)

# One shared, lazily-built sentiment engine.
engine = SentimentEngine()

# In-memory conversation memory: session_id -> deque of {role, content} turns.
# An OrderedDict + cap keeps memory bounded on a single-process demo server.
_MAX_SESSIONS = 500
_sessions: "OrderedDict[str, Deque[Dict[str, str]]]" = OrderedDict()


def _get_session(session_id: str) -> Deque[Dict[str, str]]:
    if session_id not in _sessions:
        if len(_sessions) >= _MAX_SESSIONS:
            _sessions.popitem(last=False)  # evict oldest session
        _sessions[session_id] = deque(maxlen=settings.history_turns * 2)
    else:
        _sessions.move_to_end(session_id)
    return _sessions[session_id]


# ─────────────────────────────────────────────────────────────────────
# Static UI
# ─────────────────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the chat interface."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>UI not found</h1><p>Missing static/index.html</p>", status_code=404)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    icon = STATIC_DIR / "favicon.svg"
    if icon.exists():
        return FileResponse(icon)
    raise HTTPException(status_code=404, detail="No favicon")


# ─────────────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report app, Ollama and sentiment-backend status."""
    return HealthResponse(
        status="ok",
        ollama=await llama.health(),
        sentiment_backend=engine.describe(),
        model=llama.model,
    )


@app.post("/sentiment", response_model=SentimentResult)
async def sentiment_only(payload: SentimentOnlyRequest) -> SentimentResult:
    """Sentiment detection on its own — no LLM call.

    Handy for grading accuracy independently of the chatbot.
    """
    return engine.analyse(payload.message)


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """Full pipeline: detect sentiment, then generate an appropriate reply."""
    started = time.perf_counter()
    session_id = payload.session_id or uuid.uuid4().hex

    # 1. Sentiment analysis
    sentiment = engine.analyse(payload.message)

    # 2. If the caller only wants detection, stop here.
    if not payload.generate_reply:
        return ChatResponse(
            reply="",
            sentiment=sentiment,
            session_id=session_id,
            model=llama.model,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # 3. Generate the sentiment-aware reply, with prior turns as context.
    history = list(_get_session(session_id)) if payload.session_id else []

    try:
        reply = await llama.chat(payload.message, sentiment, history=history)
    except LlamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 4. Store the turn for follow-up messages.
    memory = _get_session(session_id)
    memory.append({"role": "user", "content": payload.message})
    memory.append({"role": "assistant", "content": reply})

    return ChatResponse(
        reply=reply,
        sentiment=sentiment,
        session_id=session_id,
        model=llama.model,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/reset")
async def reset(payload: ResetRequest) -> Dict[str, str]:
    """Forget a session's conversation history."""
    _sessions.pop(payload.session_id, None)
    return {"status": "cleared", "session_id": payload.session_id}


@app.get("/api/policy")
async def policy() -> Dict[str, str]:
    """Expose the per-sentiment response policy (used in the report/UI)."""
    return SENTIMENT_POLICY


@app.get("/api/demo")
async def demo() -> Dict[str, object]:
    """The three assignment examples, with detection pre-computed."""
    examples: List[Dict[str, object]] = []
    cases = [
        ("Your support was very helpful. Thank you!", "POSITIVE"),
        ("Can you tell me my order status?", "NEUTRAL"),
        ("I am very disappointed. My order has been delayed for 5 days!", "NEGATIVE"),
    ]
    for message, expected in cases:
        result = engine.analyse(message)
        examples.append(
            {
                "user_message": message,
                "expected": expected,
                "detected": result.label,
                "confidence": result.confidence,
                "scores": result.scores,
                "intensity": result.intensity,
                "signals": result.signals,
                "correct": result.label == expected,
            }
        )
    return {
        "sentiment_backend": engine.describe(),
        "model": llama.model,
        "examples": examples,
    }


@app.get("/api/selfcheck")
async def selfcheck() -> Dict[str, object]:
    """Runtime sanity check of the sentiment stage (no LLM needed)."""
    probes = [
        ("Thank you, your service is amazing!", "POSITIVE"),
        ("Where is my order?", "NEUTRAL"),
        ("Your service is very bad. My order is late!", "NEGATIVE"),
        ("I love your service!", "POSITIVE"),
        ("I am extremely unhappy with your service.", "NEGATIVE"),
    ]
    results = []
    correct = 0
    for text, expected in probes:
        result = engine.analyse(text)
        ok = result.label == expected
        correct += int(ok)
        results.append(
            {
                "message": text,
                "expected": expected,
                "detected": result.label,
                "confidence": result.confidence,
                "correct": ok,
            }
        )
    return {
        "backend": engine.describe(),
        "accuracy": round(correct / len(probes), 4),
        "cases": results,
    }


@app.post("/api/compare")
async def compare(payload: ChatRequest) -> Dict[str, object]:
    """Demonstrate the assignment's satisfaction criterion.

    Runs the *same* message through a "normal" chatbot prompt (no sentiment
    context) and the sentiment-aware prompt, then rubric-scores both.
    Returns the two replies so the difference is visible.
    """
    sentiment = engine.analyse(payload.message)

    # Sentiment-aware reply (real LLM call).
    aware_reply = await llama.chat(payload.message, sentiment)

    # Baseline: neutral, generic support reply with no sentiment guidance.
    from .prompts import BASE_PERSONA
    from .schemas import SentimentResult as SR

    neutral = SR(
        label="NEUTRAL",
        confidence=1.0,
        scores={"POSITIVE": 0.0, "NEUTRAL": 1.0, "NEGATIVE": 0.0},
        backend="baseline",
        signals=[],
        intensity=0.0,
    )
    baseline_prompt = (
        BASE_PERSONA
        + "\n\nRespond factually to the customer. Keep the reply short."
    )

    import httpx

    payload_body = {
        "model": llama.model,
        "messages": [
            {"role": "system", "content": baseline_prompt},
            {"role": "user", "content": payload.message},
        ],
        "stream": False,
        "options": {"temperature": 0.6, "num_predict": 300},
    }
    try:
        async with httpx.AsyncClient(timeout=llama.timeout) as client:
            resp = await client.post(f"{llama.base_url}/api/chat", json=payload_body)
            resp.raise_for_status()
            baseline_reply = (resp.json().get("message") or {}).get("content", "").strip()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "message": payload.message,
        "sentiment": sentiment.model_dump(),
        "sentiment_aware": {
            "reply": aware_reply,
            "appropriate": response_is_appropriate(sentiment, aware_reply),
        },
        "baseline": {
            "reply": baseline_reply,
            "appropriate": response_is_appropriate(sentiment, baseline_reply),
        },
        "neutral_probe": neutral.label,  # kept for clarity in the response
    }


def main() -> None:
    """Entry point: ``python -m app.main``."""
    import uvicorn

    print("─" * 64)
    print("  Sentiment-Aware Customer Service Chatbot")
    print(f"  Sentiment backend : {engine.describe()}")
    if engine.transformer_error:
        print(f"  (transformer unavailable → {engine.transformer_error})")
    print(f"  Llama 3 model     : {llama.model} @ {llama.base_url}")
    print(f"  Open              : http://localhost:{settings.port}")
    print("─" * 64)

    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
