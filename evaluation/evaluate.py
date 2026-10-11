#!/usr/bin/env python3
"""Evaluation harness for the sentiment-aware chatbot.

Reports the assignment's three criteria:

  A. Accuracy of sentiment detection
       — overall accuracy, macro-F1 and a per-class classification report.

  B. Appropriateness of responses
       — every test message is paired with an expected-style reply, scored by
         the deterministic rubric in ``rubric.py``. With ``--with-llm`` the real
         Llama 3 replies are generated and scored instead.

  C. Impact on customer satisfaction
       — the same messages are answered twice: once with the sentiment-aware
         policy and once with a generic baseline prompt. The rubric scores are
         compared to quantify the improvement.

Usage
-----
    python evaluation/evaluate.py                # A + B (offline, no LLM)
    python evaluation/evaluate.py --with-llm     # A + B + C with real Llama 3
    python evaluation/evaluate.py --json out.json
    python evaluation/evaluate.py --save-baseline # snapshot results as the baseline
    python evaluation/evaluate.py --compare-baseline

Metric thresholds are enforced with ``--min-accuracy`` / ``--min-appropriate``
so the script can be used as a CI gate (non-zero exit on regression).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.sentiment import SentimentEngine  # noqa: E402
from evaluation.rubric import satisfaction_delta, score_response  # noqa: E402

TEST_SET = Path(__file__).resolve().parent / "test_set.json"
BASELINE_FILE = Path(__file__).resolve().parent / "baseline_results.json"

# Expected-style reference replies per sentiment: what a *correct* answer looks
# like. The rubric scores these to prove the rubric itself is calibrated.
REFERENCE_REPLIES: Dict[str, str] = {
    "POSITIVE": "You're very welcome — I'm really glad I could help! Is there anything else I can do for you?",
    "NEUTRAL": "Sure. Please share your order ID and I'll check the status for you.",
    "NEGATIVE": (
        "I'm sorry about this — I understand how frustrating it must be. "
        "Let me help you resolve it right away: please share your order ID and I'll investigate."
    ),
}

# Deliberately wrong replies, used to verify the rubric actually rejects them.
WRONG_REPLIES: Dict[str, str] = {
    "POSITIVE": "Your issue has been noted.",
    "NEUTRAL": "I'm so sorry, that sounds awful. Let me fix it!",
    "NEGATIVE": "Great! Have a nice day! 😄",
}

BAR = "─" * 66


def load_test_set() -> List[dict]:
    if not TEST_SET.exists():
        raise SystemExit(f"Test set not found: {TEST_SET}")
    return json.loads(TEST_SET.read_text(encoding="utf-8"))


# ── Criterion A ──────────────────────────────────────────────────────


def evaluate_accuracy(engine: SentimentEngine, cases: List[dict]) -> dict:
    y_true: List[str] = []
    y_pred: List[str] = []
    details: List[dict] = []

    for case in cases:
        result = engine.analyse(case["message"])
        y_true.append(case["label"])
        y_pred.append(result.label)
        details.append(
            {
                "message": case["message"],
                "expected": case["label"],
                "predicted": result.label,
                "confidence": result.confidence,
                "correct": result.label == case["label"],
                "signals": result.signals,
            }
        )

    total = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / total if total else 0.0

    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score

        labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
        report = classification_report(
            y_true, y_pred, labels=labels, zero_division=0, digits=3
        )
        matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    except ImportError:
        macro_f1 = float("nan")
        report = "(install scikit-learn for the full report)"
        matrix = None

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": macro_f1,
        "correct": correct,
        "total": total,
        "report": report,
        "confusion_matrix": matrix,
        "details": details,
        "errors": [d for d in details if not d["correct"]],
    }


# ── Criterion B ──────────────────────────────────────────────────────


def evaluate_appropriateness(cases: List[dict]) -> dict:
    """Score the reference replies, and confirm the rubric rejects bad ones."""
    per_class: Dict[str, List[float]] = {}
    rubric_failures: List[dict] = []

    for case in cases:
        label = case["label"]
        reply = REFERENCE_REPLIES[label]
        result = score_response(label, reply)
        per_class.setdefault(label, []).append(result.score)
        if not result.appropriate:
            rubric_failures.append(
                {"message": case["message"], "reply": reply, "reasons": result.reasons}
            )

    class_scores = {k: round(sum(v) / len(v), 3) for k, v in per_class.items()}
    overall = round(sum(class_scores.values()) / len(class_scores), 3) if class_scores else 0.0

    # Rubric sanity check: correct answers must pass, wrong answers must fail.
    sanity = []
    for label, wrong in WRONG_REPLIES.items():
        res = score_response(label, wrong)
        sanity.append(
            {
                "sentiment": label,
                "wrong_reply": wrong,
                "rejected": not res.appropriate,
                "score": res.score,
                "reasons": res.reasons,
            }
        )

    return {
        "mean_appropriateness": overall,
        "per_sentiment": class_scores,
        "reference_failures": rubric_failures,
        "rubric_sanity": sanity,
    }


# ── Criterion C ──────────────────────────────────────────────────────


async def evaluate_satisfaction(cases: List[dict], engine: SentimentEngine) -> dict:
    """Compare sentiment-aware vs baseline replies using real Llama 3."""
    from app.llm import LlamaError, llama
    from app.prompts import BASE_PERSONA

    import httpx

    health = await llama.health()
    if health != "ok":
        raise SystemExit(
            f"Llama 3 is not available (status: {health}).\n"
            f"Start it with `ollama serve` and pull the model with "
            f"`ollama pull {llama.model}`, then retry."
        )

    # One representative message per sentiment keeps the run short.
    sample = []
    for label in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
        match = next((c for c in cases if c["label"] == label), None)
        if match:
            sample.append(match)

    rows = []
    for case in sample:
        sentiment = engine.analyse(case["message"])

        aware = await llama.chat(case["message"], sentiment)

        baseline_prompt = BASE_PERSONA + "\n\nRespond factually to the customer. Keep it short."
        payload = {
            "model": llama.model,
            "messages": [
                {"role": "system", "content": baseline_prompt},
                {"role": "user", "content": case["message"]},
            ],
            "stream": False,
            "options": {"temperature": 0.6, "num_predict": 300},
        }
        async with httpx.AsyncClient(timeout=llama.timeout) as client:
            resp = await client.post(f"{llama.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            baseline = (resp.json().get("message") or {}).get("content", "").strip()

        delta = satisfaction_delta(sentiment.label, aware, baseline)
        delta["message"] = case["message"]
        delta["detected"] = sentiment.label
        delta["sentiment_aware_reply"] = aware
        delta["baseline_reply"] = baseline
        rows.append(delta)

    mean_delta = round(sum(r["delta"] for r in rows) / len(rows), 3) if rows else 0.0
    return {"rows": rows, "mean_delta": mean_delta}


# ── Baseline persistence & regression gating ─────────────────────────


def load_baseline() -> dict:
    if not BASELINE_FILE.exists():
        raise SystemExit(
            f"No baseline found at {BASELINE_FILE}. "
            "Create one first with `python evaluation/evaluate.py --save-baseline`."
        )
    return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))


def save_baseline(results: dict) -> None:
    BASELINE_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Baseline saved to {BASELINE_FILE}")


def compare_to_baseline(results: dict, baseline: dict) -> dict:
    """Diff the headline metrics of two runs (higher is better for all)."""
    pairs = [
        ("accuracy", baseline["accuracy"]["accuracy"], results["accuracy"]["accuracy"]),
        (
            "macro_f1",
            baseline["accuracy"].get("macro_f1", float("nan")),
            results["accuracy"].get("macro_f1", float("nan")),
        ),
        (
            "appropriateness",
            baseline["appropriateness"]["mean_appropriateness"],
            results["appropriateness"]["mean_appropriateness"],
        ),
    ]

    rows = []
    for name, old, new in pairs:
        if old != old or new != new:  # NaN guard
            continue
        rows.append(
            {"metric": name, "baseline": old, "current": new, "delta": round(new - old, 4)}
        )
    return {"rows": rows}


def print_comparison(cmp: dict) -> None:
    print(BAR)
    print("REGRESSION CHECK vs BASELINE")
    print(BAR)
    for row in cmp["rows"]:
        if row["delta"] > 0:
            mark = "✅ improved"
        elif row["delta"] < 0:
            mark = "❌ REGRESSED"
        else:
            mark = "➖ unchanged"
        print(f"  {row['metric']:<16} {row['baseline']:.4f} → {row['current']:.4f} "
              f"({row['delta']:+.4f})  {mark}")
    print()


def check_thresholds(results: dict, min_accuracy: float, min_appropriate: float) -> List[str]:
    """Return a list of threshold violations (empty means the gate passed)."""
    failures: List[str] = []
    acc = results["accuracy"]["accuracy"]
    app = results["appropriateness"]["mean_appropriateness"]
    if acc < min_accuracy:
        failures.append(f"accuracy {acc:.4f} < required {min_accuracy:.4f}")
    if app < min_appropriate:
        failures.append(f"appropriateness {app:.4f} < required {min_appropriate:.4f}")
    return failures


# ── Reporting ────────────────────────────────────────────────────────


def print_accuracy(res: dict) -> None:
    print(BAR)
    print("A. ACCURACY OF SENTIMENT DETECTION")
    print(BAR)
    print(f"  Correct : {res['correct']} / {res['total']}")
    print(f"  Accuracy: {res['accuracy']:.4f}")
    if res["macro_f1"] == res["macro_f1"]:  # not NaN
        print(f"  Macro-F1: {res['macro_f1']:.4f}")
    print()
    print(res["report"])

    if res["confusion_matrix"]:
        labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        print("  Confusion matrix (rows = expected, cols = predicted)")
        print(f"  {'':<10}" + "".join(f"{l:>11}" for l in labels))
        for label, row in zip(labels, res["confusion_matrix"]):
            print(f"  {label:<10}" + "".join(f"{v:>11}" for v in row))
        print()

    if res["errors"]:
        print(f"  Misclassified ({len(res['errors'])}):")
        for err in res["errors"]:
            print(f"    expected {err['expected']:<8} got {err['predicted']:<8} "
                  f"({err['confidence']:.2f})  \"{err['message'][:52]}\"")
    else:
        print("  No misclassifications. 🎉")
    print()


def print_appropriateness(res: dict) -> None:
    print(BAR)
    print("B. APPROPRIATENESS OF RESPONSES")
    print(BAR)
    print(f"  Mean appropriateness: {res['mean_appropriateness']:.3f}")
    for label, score in res["per_sentiment"].items():
        print(f"    {label:<10} {score:.3f}")
    print()

    print("  Rubric sanity check (wrong replies must be rejected):")
    for item in res["rubric_sanity"]:
        mark = "✅ rejected" if item["rejected"] else "❌ ACCEPTED"
        print(f"    {item['sentiment']:<10} score {item['score']:.2f}  {mark}")
        print(f"        \"{item['wrong_reply'][:60]}\"")
    print()


def print_satisfaction(res: dict) -> None:
    print(BAR)
    print("C. IMPACT ON CUSTOMER SATISFACTION (sentiment-aware vs baseline)")
    print(BAR)
    for row in res["rows"]:
        print(f"  [{row['sentiment']}] \"{row['message'][:56]}\"")
        print(f"    baseline score        : {row['baseline_score']:.2f}")
        print(f"    sentiment-aware score : {row['sentiment_aware_score']:.2f}"
              f"   (delta {row['delta']:+.2f})")
        print(f"    baseline reply        : {row['baseline_reply'][:110]}")
        print(f"    sentiment-aware reply : {row['sentiment_aware_reply'][:110]}")
        print()
    print(f"  Mean improvement: {res['mean_delta']:+.3f}")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the sentiment-aware chatbot.")
    parser.add_argument("--with-llm", action="store_true",
                        help="Also run criterion C with real Llama 3 replies.")
    parser.add_argument("--backend", default=None,
                        help="Force a sentiment backend: auto | transformer | lexicon")
    parser.add_argument("--json", default=None, help="Write the full results to this JSON file.")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Store this run's results as the comparison baseline.")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="Diff this run against the saved baseline.")
    parser.add_argument("--min-accuracy", type=float, default=None,
                        help="Fail (exit 1) if sentiment accuracy drops below this.")
    parser.add_argument("--min-appropriate", type=float, default=None,
                        help="Fail (exit 1) if mean appropriateness drops below this.")
    args = parser.parse_args()

    engine = SentimentEngine(backend=args.backend)
    cases = load_test_set()

    print()
    print(BAR)
    print("  SENTIMENT-AWARE CHATBOT — EVALUATION REPORT")
    print(BAR)
    print(f"  Sentiment backend : {engine.describe()}")
    print(f"  Test set          : {len(cases)} labelled messages")
    if engine.transformer_error:
        print(f"  Note              : transformer unavailable → {engine.transformer_error}")
    print()

    accuracy = evaluate_accuracy(engine, cases)
    appropriateness = evaluate_appropriateness(cases)

    print_accuracy(accuracy)
    print_appropriateness(appropriateness)

    results: dict = {"accuracy": accuracy, "appropriateness": appropriateness}

    if args.compare_baseline:
        comparison = compare_to_baseline(results, load_baseline())
        results["comparison"] = comparison
        print_comparison(comparison)

    if args.with_llm:
        satisfaction = await evaluate_satisfaction(cases, engine)
        results["satisfaction"] = satisfaction
        print_satisfaction(satisfaction)
    else:
        print(BAR)
        print("C. IMPACT ON CUSTOMER SATISFACTION")
        print(BAR)
        print("  Skipped (offline mode). Run with --with-llm to compare real")
        print("  sentiment-aware replies against a generic baseline.")
        print()

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Full results written to {out}")

    if args.save_baseline:
        save_baseline(results)

    failures: List[str] = []
    if args.min_accuracy is not None or args.min_appropriate is not None:
        failures = check_thresholds(
            results,
            args.min_accuracy if args.min_accuracy is not None else 0.0,
            args.min_appropriate if args.min_appropriate is not None else 0.0,
        )
        if failures:
            print(BAR)
            print("THRESHOLD GATE: FAILED")
            for failure in failures:
                print(f"  ✗ {failure}")
            print()

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
