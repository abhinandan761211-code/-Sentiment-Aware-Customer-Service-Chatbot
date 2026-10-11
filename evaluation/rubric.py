"""Response-appropriateness rubric.

Implements the assignment's "Appropriateness of responses" criterion as a
deterministic, reproducible check — no human grader required. It is deliberately
conservative: it looks for *evidence* that the reply matches the sentiment, and
for the specific failure modes described in the brief (cheerful reply to an
angry customer, apologetic reply to a happy one, over-emotional reply to a
neutral question).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

APOLOGY = ("sorry", "apolog", "regret")
EMPATHY = ("understand", "frustrat", "appreciate your patience", "i can imagine")
ACTION = (
    "help", "resolve", "fix", "check", "look into", "escalate", "investigate",
    "refund", "replace", "assist", "share your", "provide your", "let me",
    "i'll", "i will", "arrange", "priorit",
)
INFO_REQUEST = (
    "order id", "order number", "provide", "share", "let me know", "details",
    "email", "reference number",
)
ACK = (
    "welcome", "thank you", "thanks", "glad", "happy to hear", "pleasure",
    "appreciate", "delighted", "my pleasure", "so glad",
)
GENERIC = (
    "noted", "your issue has been", "we will get back", "have a nice day",
    "ok", "okay", "got it",
)
CHEERFUL = ("great!", "have a nice day", "awesome!", "wonderful!", "enjoy", "😄", "🎉")
ACTION_EMOJI = ("😊", "🙂", "😍")


@dataclass
class AppropriatenessResult:
    appropriate: bool
    score: float
    reasons: List[str] = field(default_factory=list)


def _hits(text: str, needles) -> List[str]:
    return [n for n in needles if n in text]


def score_response(sentiment_label: str, reply: str) -> AppropriatenessResult:
    """Score one reply against the policy for the detected sentiment (0..1)."""
    text = (reply or "").lower().strip()
    if not text:
        return AppropriatenessResult(False, 0.0, ["empty reply"])

    reasons: List[str] = []
    score = 0.0

    if sentiment_label == "NEGATIVE":
        empathy = _hits(text, APOLOGY + EMPATHY)
        action = _hits(text, ACTION + INFO_REQUEST)
        cheerful = _hits(text, CHEERFUL)

        if empathy:
            score += 0.4
            reasons.append(f"empathy present ({empathy[0]})")
        else:
            reasons.append("MISSING empathy/apology")

        if action:
            score += 0.4
            reasons.append(f"concrete action ({action[0]})")
        else:
            reasons.append("MISSING concrete next step")

        if not cheerful:
            score += 0.2
            reasons.append("tone appropriate")
        else:
            reasons.append(f"WRONG cheerful tone ({cheerful[0]})")

    elif sentiment_label == "POSITIVE":
        ack = _hits(text, ACK)
        apology = _hits(text, APOLOGY)

        if ack:
            score += 0.6
            reasons.append(f"acknowledges gratitude ({ack[0]})")
        else:
            reasons.append("MISSING acknowledgement")

        if not apology:
            score += 0.2
            reasons.append("no inappropriate apology")
        else:
            reasons.append("WRONG apologetic tone for a happy customer")

        if len(text) <= 400:
            score += 0.2
            reasons.append("appropriately brief")

    else:  # NEUTRAL
        informative = _hits(text, ("can", "could", "please", "provide", "let me",
                                   "help", "status", "sure", "of course", "here",
                                   "order id", "details", "will", "you can"))
        over_emotional = _hits(text, APOLOGY + CHEERFUL + ("amazing", "terrible", "awful"))

        if informative:
            score += 0.6
            reasons.append(f"informative ({informative[0]})")
        else:
            reasons.append("not clearly informative")

        if not over_emotional:
            score += 0.4
            reasons.append("neutral tone preserved")
        else:
            reasons.append(f"WRONG emotional tone ({over_emotional[0]})")

    appropriate = score >= 0.7
    return AppropriatenessResult(appropriate, round(score, 3), reasons)


def satisfaction_delta(sentiment_label: str, aware_reply: str, baseline_reply: str) -> dict:
    """Proxy for the assignment's 'impact on customer satisfaction' criterion.

    Instead of real customers we measure the observable difference a
    sentiment-aware reply makes: how many of the policy's required behaviours
    it satisfies that the generic baseline does not. Returns a dict with both
    scores and the delta so the improvement is reportable as a number.
    """
    aware = score_response(sentiment_label, aware_reply)
    base = score_response(sentiment_label, baseline_reply)

    return {
        "sentiment": sentiment_label,
        "sentiment_aware_score": aware.score,
        "baseline_score": base.score,
        "delta": round(aware.score - base.score, 3),
        "sentiment_aware_reasons": aware.reasons,
        "baseline_reasons": base.reasons,
    }
