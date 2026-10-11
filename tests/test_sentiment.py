"""Tests for the sentiment engine, prompt builder and rubric.

Run with:  python -m pytest tests/ -v       (or)  python tests/test_sentiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.prompts import build_system_prompt, build_user_prompt  # noqa: E402
from app.sentiment import SentimentEngine, response_is_appropriate  # noqa: E402
from evaluation.rubric import score_response  # noqa: E402

ENGINE = SentimentEngine(backend="lexicon")


# ── sentiment detection ──────────────────────────────────────────────


def test_positive_examples():
    for msg in [
        "Thank you, your service is amazing!",
        "Your support was very helpful. Thank you!",
        "I love your service!",
        "Great job, my order arrived early.",
        "Kudos to the team for the quick delivery!",
    ]:
        assert ENGINE.analyse(msg).label == "POSITIVE", msg


def test_negative_examples():
    for msg in [
        "Your service is very bad. My order is late!",
        "I am very disappointed. My order has been delayed for 5 days!",
        "I have been waiting for my refund for 10 days. This is ridiculous!",
        "Your support was rude and unhelpful.",
        "Worst experience ever, I want to cancel everything.",
    ]:
        assert ENGINE.analyse(msg).label == "NEGATIVE", msg


def test_neutral_examples():
    for msg in [
        "Where is my order?",
        "Can you tell me my order status?",
        "What is your return policy?",
        "How do I change my delivery address?",
        "Please share the tracking details.",
    ]:
        assert ENGINE.analyse(msg).label == "NEUTRAL", msg


def test_negation_handling():
    """The classic failure mode: 'not good' / 'not bad'."""
    assert ENGINE.analyse("The service was not good at all.").label == "NEGATIVE"
    assert ENGINE.analyse("Not bad, quite good actually.").label == "POSITIVE"
    assert ENGINE.analyse("I am not happy with this.").label == "NEGATIVE"


def test_intensifiers_raise_confidence():
    plain = ENGINE.analyse("This is bad.")
    strong = ENGINE.analyse("This is extremely very bad!!!")
    assert strong.confidence >= plain.confidence
    assert strong.intensity >= plain.intensity


def test_shouting_is_negative():
    """A neutral sentence shouted in caps should not stay neutral."""
    result = ENGINE.analyse("I WANT MY MONEY BACK NOW!!!")
    assert result.label == "NEGATIVE"
    assert result.intensity > 0.3


def test_scores_are_a_distribution():
    result = ENGINE.analyse("Thank you so much, this was great!")
    assert abs(sum(result.scores.values()) - 1.0) < 0.02
    assert 0.0 <= result.confidence <= 1.0
    assert result.label in result.scores


def test_empty_message_is_neutral():
    result = ENGINE.analyse("   ")
    assert result.label == "NEUTRAL"
    assert result.confidence == 1.0


def test_signals_are_reported():
    result = ENGINE.analyse("I am very disappointed with the delay.")
    assert result.signals, "expected explainability signals"


# ── prompt building ──────────────────────────────────────────────────


def test_prompt_differs_per_sentiment():
    positives = [
        build_system_prompt(ENGINE.analyse("Amazing service, thank you!")),
        build_system_prompt(ENGINE.analyse("Where is my order?")),
        build_system_prompt(ENGINE.analyse("This is terrible and unacceptable!")),
    ]
    assert len(set(positives)) == 3, "each sentiment must produce a distinct policy"

    assert "POSITIVE" in positives[0]
    assert "NEUTRAL" in positives[1]
    assert "NEGATIVE" in positives[2]


def test_negative_prompt_forbids_cheerfulness():
    prompt = build_system_prompt(ENGINE.analyse("I am extremely unhappy with your service."))
    assert "never be cheerful" in prompt.lower()
    assert "apolog" in prompt.lower()


def test_user_prompt_keeps_message_verbatim():
    msg = "My order #48213 is late!"
    sentiment = ENGINE.analyse(msg)
    prompt = build_user_prompt(msg, sentiment)
    assert msg in prompt
    assert sentiment.label in prompt
    assert "sentiment_context" in prompt


def test_high_intensity_adds_escalation_note():
    calm = build_system_prompt(ENGINE.analyse("This was a bit disappointing."))
    loud = build_system_prompt(
        ENGINE.analyse("THIS IS RIDICULOUS!!! UNACCEPTABLE!!! I WANT A REFUND NOW!!!")
    )
    assert "escalation note" in loud.lower()
    assert "escalation note" not in calm.lower()


# ── response appropriateness ─────────────────────────────────────────


def test_negative_reply_without_empathy_is_rejected():
    sentiment = ENGINE.analyse("I am extremely unhappy with your service.")
    assert not response_is_appropriate(sentiment, "Great! Have a nice day! 😄")
    assert not score_response("NEGATIVE", "Great! Have a nice day! 😄").appropriate


def test_appropriate_replies_pass():
    good = {
        "POSITIVE": "You're very welcome! I'm glad I could help. Is there anything else?",
        "NEUTRAL": "Sure. Please provide your order ID so I can check the status.",
        "NEGATIVE": (
            "I'm sorry for the delay — I understand how frustrating this must be. "
            "Please share your order ID and I'll help you resolve this."
        ),
    }
    for label, reply in good.items():
        assert response_is_appropriate(_mk(label), reply), label
        assert score_response(label, reply).appropriate, label


def _mk(label: str):
    """Build a SentimentResult-like object with the given label."""
    from app.schemas import SentimentResult

    scores = {"POSITIVE": 0.0, "NEUTRAL": 0.0, "NEGATIVE": 0.0}
    scores[label] = 1.0
    return SentimentResult(
        label=label, confidence=1.0, scores=scores, backend="test", signals=[], intensity=0.0
    )


# ── accuracy smoke test ──────────────────────────────────────────────


def test_accuracy_on_labelled_set():
    import json

    cases = json.loads((ROOT / "evaluation" / "test_set.json").read_text(encoding="utf-8"))
    correct = sum(1 for c in cases if ENGINE.analyse(c["message"]).label == c["label"])
    accuracy = correct / len(cases)
    # The lexicon model must clear a sensible bar; the transformer does better.
    assert accuracy >= 0.80, f"lexicon accuracy too low: {accuracy:.3f}"


if __name__ == "__main__":
    import traceback

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
