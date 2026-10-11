"""Sentiment analysis engine.

Two interchangeable backends behind one interface:

* ``transformer`` — a RoBERTa model fine-tuned on social text
  (``cardiffnlp/twitter-roberta-base-sentiment-latest``). Highest accuracy, but
  needs ``transformers`` + ``torch`` and a one-off model download.
* ``lexicon`` — a transparent, dependency-free rule/lexicon model with negation
  and intensifier handling. Always available, so the whole project runs on a
  machine with no ML stack at all.

Both return the same three-class result: POSITIVE / NEUTRAL / NEGATIVE.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import settings
from .schemas import SentimentResult

LABELS = ("NEGATIVE", "NEUTRAL", "POSITIVE")

# ─────────────────────────────────────────────────────────────────────
# Lexicon backend
# ─────────────────────────────────────────────────────────────────────

# Weights are hand-tuned: 1.0 = mildly positive/negative, 2.0+ = strong.
POSITIVE_WORDS: Dict[str, float] = {
    # service praise
    "amazing": 2.0, "awesome": 2.0, "excellent": 2.0, "fantastic": 2.0,
    "perfect": 2.0, "wonderful": 2.0, "superb": 2.0, "brilliant": 1.8,
    "great": 1.6, "good": 1.2, "nice": 1.2, "helpful": 1.8, "helpfully": 1.6,
    "thank": 1.6, "thanks": 1.6, "thankyou": 1.6, "grateful": 1.8,
    "appreciate": 1.7, "appreciated": 1.7, "love": 2.0, "loved": 2.0,
    "like": 0.9, "happy": 1.8, "glad": 1.6, "pleased": 1.6, "delighted": 1.9,
    "satisfied": 1.5, "impressed": 1.7, "impressive": 1.7, "recommend": 1.5,
    "recommended": 1.5, "responsive": 1.4, "quick": 1.0, "fast": 1.0,
    "smooth": 1.3, "easy": 1.1, "kind": 1.4, "polite": 1.4, "friendly": 1.5,
    "professional": 1.3, "patient": 1.2, "supportive": 1.6, "courteous": 1.4,
    "best": 1.8, "worth": 1.2, "worthwhile": 1.3, "yes": 0.5, "sure": 0.4,
    "resolved": 1.6, "fix": 0.6, "fixed": 1.5, "working": 0.8, "works": 0.8,
    "legit": 1.0, "kudos": 1.8, "bravo": 1.8, "cheers": 1.4, "yay": 1.6,
}

NEGATIVE_WORDS: Dict[str, float] = {
    # complaints
    "bad": 1.6, "terrible": 2.0, "awful": 2.0, "horrible": 2.0, "worst": 2.2,
    "hate": 2.0, "hated": 2.0, "disappointed": 2.0, "disappointing": 2.0,
    "disappointment": 2.0, "unhappy": 1.9, "frustrated": 1.9,
    "frustrating": 1.9, "frustration": 1.8, "angry": 2.0, "annoyed": 1.6,
    "annoying": 1.6, "upset": 1.7, "mad": 1.7, "ridiculous": 2.0,
    "unacceptable": 2.1, "useless": 2.0, "pathetic": 2.1, "garbage": 2.0,
    "joke": 1.6, "scam": 2.2, "fraud": 2.2, "cheat": 1.9, "cheated": 1.9,
    "broken": 1.7, "damaged": 1.5, "defective": 1.7, "faulty": 1.6,
    "delay": 1.5, "delayed": 1.5, "late": 1.4, "slow": 1.3, "waiting": 0.9,
    "waited": 1.0, "never": 1.2, "still": 0.6, "again": 0.6, "problem": 1.3,
    "problems": 1.4, "issue": 1.1, "issues": 1.2, "error": 1.2, "wrong": 1.4,
    "fail": 1.6, "failed": 1.7, "failure": 1.7, "worse": 1.7, "poor": 1.6,
    "rude": 2.0, "ignored": 1.8, "ignore": 1.6, "unhelpful": 1.9,
    "careless": 1.6, "negligent": 1.8, "incompetent": 1.9, "mess": 1.5,
    "refund": 0.6, "complaint": 1.6, "complain": 1.5, "disgusted": 2.1,
    "furious": 2.2, "outrageous": 2.0, "sick": 1.4, "tired": 1.5,
    "stuck": 1.4, "lost": 1.2, "missing": 1.3, "cancelled": 1.2,
    "canceled": 1.2, "charged": 0.8, "overcharged": 1.9,
    # NOTE: negators like "no"/"not" are intentionally NOT scored here —
    # they are handled by the negation logic below. Scoring them here would
    # double-count ("no" would be both a negator and a negative word).
}

# Words that flip the polarity of the following window.
NEGATORS = {
    "not", "no", "never", "none", "cannot", "can't", "cant", "won't", "wont",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "isn't", "isnt",
    "aren't", "arent", "wasn't", "wasnt", "weren't", "werent", "without",
    "hardly", "barely", "lack", "lacks", "lacking", "n't",
}

# Modifiers that scale the next sentiment word.
INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "extremely": 1.8, "so": 1.3, "too": 1.3,
    "incredibly": 1.7, "absolutely": 1.7, "totally": 1.5, "highly": 1.5,
    "super": 1.5, "deeply": 1.5, "seriously": 1.4, "completely": 1.6,
    "utterly": 1.7, "awfully": 1.6, "terribly": 1.6,
}

# Words that signal the customer is escalating / close to churning.
ESCALATION_WORDS = {
    "ridiculous", "unacceptable", "furious", "lawyer", "legal", "sue",
    "cancel", "cancelling", "canceling", "refund", "never again", "scam",
    "fraud", "worst", "disgusted", "ashamed", "complaint", "escalate",
    "manager", "supervisor",
}

APOLOGY_MARKERS = {"sorry", "apologize", "apologies", "regret"}

TOKEN_RE = re.compile(r"[a-z0-9']+")
EXCLAIM_RE = re.compile(r"!{1,}")
QUESTION_RE = re.compile(r"\?")


@dataclass
class _LexiconScore:
    label: str
    confidence: float
    scores: Dict[str, float]
    signals: List[str] = field(default_factory=list)
    intensity: float = 0.0


def _normalise(text: str) -> str:
    """Lowercase but keep the raw text for punctuation-based intensity."""
    return text.lower()


def _compute_intensity(raw: str) -> float:
    """How strongly the emotion is expressed, 0..1.

    Driven by: ratio of ALL-CAPS words, exclamation runs, repeated punctuation
    and the presence of escalation vocabulary.
    """
    words = [w for w in re.findall(r"[A-Za-z]+", raw) if len(w) > 2]
    caps_ratio = 0.0
    if words:
        caps_ratio = sum(1 for w in words if w.isupper()) / len(words)

    exclaims = len(EXCLAIM_RE.findall(raw))
    bang_score = min(exclaims / 3.0, 1.0)

    lower = _normalise(raw)
    escalation = 1.0 if any(w in lower for w in ESCALATION_WORDS) else 0.0

    intensity = 0.45 * caps_ratio + 0.3 * bang_score + 0.25 * escalation
    return round(min(max(intensity, 0.0), 1.0), 3)


def _lexicon_predict(text: str) -> _LexiconScore:
    """Rule + lexicon sentiment classifier with negation and intensifiers."""
    raw = text
    lower = _normalise(text)
    tokens = TOKEN_RE.findall(lower)

    pos_hits: List[str] = []
    neg_hits: List[str] = []
    pos_score = 0.0
    neg_score = 0.0

    # Negation scope: a negator flips sentiment words within a SHORT window,
    # but the window must NOT cross a clause boundary. Without this,
    # "not bad, quite good" would wrongly negate "good".
    raw_lower = raw.lower()
    clauses: List[int] = []      # clause id per token
    clause_id = 0
    cursor = 0
    for match in TOKEN_RE.finditer(raw_lower):
        separator = raw_lower[cursor:match.start()]
        if any(ch in separator for ch in ",;.?!"):
            clause_id += 1
        clauses.append(clause_id)
        cursor = match.end()

    negated_until = -1       # last token index still inside the negation window
    negated_clause = -1      # clause the negator lives in
    scale = 1.0

    for i, token in enumerate(tokens):
        if token in NEGATORS:
            negated_until = i + 2          # negation applies over a short window
            negated_clause = clauses[i] if i < len(clauses) else -1
            scale = 1.0
            continue
        if token in INTENSIFIERS:
            scale = INTENSIFIERS[token]
            continue

        # A sentiment word is negated only if it is inside the window AND in
        # the SAME clause as the negator — punctuation ends the negation's reach.
        is_negated = (
            negated_until >= 0
            and i <= negated_until
            and (i >= len(clauses) or clauses[i] == negated_clause)
        )

        if token in POSITIVE_WORDS and POSITIVE_WORDS[token] > 0:
            weight = POSITIVE_WORDS[token] * scale
            if is_negated:                 # "not good" → negative
                neg_score += weight * 1.1
                neg_hits.append(f"not {token}")
            else:
                pos_score += weight
                pos_hits.append(token)
        elif token in NEGATIVE_WORDS and NEGATIVE_WORDS[token] > 0:
            weight = NEGATIVE_WORDS[token] * scale
            if is_negated:                 # "not bad" → mildly positive
                pos_score += weight * 0.7
                pos_hits.append(f"not {token}")
            else:
                neg_score += weight
                neg_hits.append(token)

        scale = 1.0

    # Punctuation contributes to valence, not just intensity.
    exclaims = len(EXCLAIM_RE.findall(raw))
    if exclaims:
        if neg_score > pos_score:
            neg_score += min(exclaims * 0.4, 1.6)
        elif pos_score > neg_score:
            pos_score += min(exclaims * 0.4, 1.6)

    # "?" on its own suppresses the emotional reading → informational tone.
    questions = len(QUESTION_RE.findall(raw))

    net = pos_score - neg_score
    magnitude = pos_score + neg_score

    if magnitude < 0.8:
        # No meaningful emotional vocabulary → neutral.
        label = "NEUTRAL"
    elif abs(net) < 0.6:
        label = "NEUTRAL"
    elif net > 0:
        label = "POSITIVE"
    else:
        label = "NEGATIVE"

    # Turn the raw scores into a probability distribution.
    # Softmax-ish over (pos, neutral_prior, neg) keeps confidences comparable
    # to what a trained model reports.
    neutral_prior = max(0.9, 1.6 - magnitude * 0.35 + questions * 0.25)
    logits = {"POSITIVE": pos_score, "NEUTRAL": neutral_prior, "NEGATIVE": neg_score}
    exps = {k: math.exp(min(v, 20.0)) for k, v in logits.items()}
    total = sum(exps.values())
    scores = {k: round(v / total, 4) for k, v in exps.items()}

    intensity = _compute_intensity(raw)
    confidence = scores[label]

    signals: List[str] = []
    if label == "POSITIVE":
        signals = sorted(set(pos_hits))[:6]
    elif label == "NEGATIVE":
        signals = sorted(set(neg_hits))[:6]
    else:
        signals = sorted(set(pos_hits + neg_hits))[:4]

    return _LexiconScore(
        label=label,
        confidence=round(confidence, 4),
        scores=scores,
        signals=signals,
        intensity=intensity,
    )


# ─────────────────────────────────────────────────────────────────────
# Transformer backend
# ─────────────────────────────────────────────────────────────────────

_TRANSFORMER_LABEL_MAP = {
    "positive": "POSITIVE",
    "negative": "NEGATIVE",
    "neutral": "NEUTRAL",
    "label_0": "NEGATIVE",
    "label_1": "NEUTRAL",
    "label_2": "POSITIVE",
}


class _TransformerBackend:
    """Lazy wrapper around a HuggingFace sentiment pipeline."""

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline  # imported lazily on purpose

        self.model_name = model_name
        self._pipe = pipeline("sentiment-analysis", model=model_name, top_k=None)

    def predict(self, text: str) -> _LexiconScore:
        raw_results = self._pipe(text[:512])[0]
        scores: Dict[str, float] = {"POSITIVE": 0.0, "NEUTRAL": 0.0, "NEGATIVE": 0.0}
        for item in raw_results:
            key = _TRANSFORMER_LABEL_MAP.get(str(item["label"]).lower())
            if key:
                scores[key] = float(item["score"])

        label = max(scores, key=scores.get)
        return _LexiconScore(
            label=label,
            confidence=round(scores[label], 4),
            scores={k: round(v, 4) for k, v in scores.items()},
            signals=[],
            intensity=_compute_intensity(text),
        )


# ─────────────────────────────────────────────────────────────────────
# Public engine
# ─────────────────────────────────────────────────────────────────────


class SentimentEngine:
    """Facade that hides which backend is actually doing the work."""

    def __init__(self, backend: Optional[str] = None, model_name: Optional[str] = None) -> None:
        requested = (backend or settings.sentiment_backend).lower()
        self.model_name = model_name or settings.sentiment_model
        self._transformer: Optional[_TransformerBackend] = None
        self._transformer_error: Optional[str] = None
        self.backend = self._resolve_backend(requested)

    def _resolve_backend(self, requested: str) -> str:
        if requested == "lexicon":
            return "lexicon"
        try:
            self._transformer = _TransformerBackend(self.model_name)
            return "transformer"
        except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
            self._transformer_error = f"{type(exc).__name__}: {exc}"
            if requested == "transformer":
                raise
            return "lexicon"

    # -- diagnostics -------------------------------------------------
    @property
    def transformer_error(self) -> Optional[str]:
        """Why the transformer backend is unavailable, if it is."""
        return self._transformer_error

    def describe(self) -> str:
        if self.backend == "transformer":
            return f"transformer ({self.model_name})"
        return "lexicon (built-in rules + negations)"

    # -- main API ----------------------------------------------------
    def analyse(self, text: str) -> SentimentResult:
        text = (text or "").strip()
        if not text:
            return SentimentResult(
                label="NEUTRAL",
                confidence=1.0,
                scores={"POSITIVE": 0.0, "NEUTRAL": 1.0, "NEGATIVE": 0.0},
                backend="lexicon",
                signals=[],
                intensity=0.0,
            )

        score = (
            self._transformer.predict(text)
            if self.backend == "transformer" and self._transformer is not None
            else _lexicon_predict(text)
        )

        # Blending: a heavy use of "!" / SHOUTING pushes a neutral message into
        # negative territory, because that is what it means in customer support.
        if score.label == "NEUTRAL" and score.intensity >= 0.5:
            boosted = dict(score.scores)
            boosted["NEGATIVE"] = round(min(0.99, boosted["NEGATIVE"] + score.intensity * 0.4), 4)
            boosted["NEUTRAL"] = round(max(0.01, 1.0 - boosted["NEGATIVE"]), 4)
            score = _LexiconScore(
                label="NEGATIVE",
                confidence=boosted["NEGATIVE"],
                scores=boosted,
                signals=score.signals + ["shouting/emphasis"],
                intensity=score.intensity,
            )

        return SentimentResult(
            label=score.label,  # type: ignore[arg-type]
            confidence=score.confidence,
            scores=score.scores,
            backend=self.backend,
            signals=score.signals,
            intensity=score.intensity,
        )


# ─────────────────────────────────────────────────────────────────────
# Helpers used by the response layer
# ─────────────────────────────────────────────────────────────────────


def response_is_appropriate(sentiment: SentimentResult, reply: str) -> bool:
    """Rubric check: does this reply match the detected sentiment?

    Used by the evaluation harness and by the API's self-check. Deliberately
    simple and deterministic so it is reproducible.
    """
    text = (reply or "").lower()
    if not text:
        return False

    if sentiment.label == "NEGATIVE":
        has_empathy = any(m in text for m in APOLOGY_MARKERS) or any(
            m in text for m in ("understand", "sorry", "apolog")
        )
        has_action = any(
            m in text
            for m in (
                "help", "resolve", "fix", "check", "look into", "escalate",
                "refund", "replace", "assist", "share your", "provide your",
                "let me", "investigate",
            )
        )
        # A negative message must never be answered with cheerfulness.
        has_wrong_tone = any(m in text for m in ("great!", "have a nice day", "awesome!", "glad"))
        return has_empathy and has_action and not has_wrong_tone

    if sentiment.label == "POSITIVE":
        has_acknowledgement = any(
            m in text
            for m in (
                "welcome", "thank you", "thanks", "glad", "happy to hear",
                "pleasure", "appreciate", "delighted",
            )
        )
        has_wrong_tone = any(m in text for m in APOLOGY_MARKERS)
        return has_acknowledgement and not has_wrong_tone

    # NEUTRAL → informative, no over-emotional language.
    has_informative = any(
        m in text
        for m in (
            "can", "could", "please", "provide", "let me", "help", "status",
            "sure", "of course", "here", "sure!", "order id", "details",
        )
    )
    has_over_emotional = any(
        m in text
        for m in (*APOLOGY_MARKERS, "amazing", "terrible", "awful")
    )
    return has_informative and not has_over_emotional
