# ──────────────────────────────────────────────────────────────
#  Sentiment-Aware Customer Service Chatbot
# ──────────────────────────────────────────────────────────────

A customer-service chatbot that **detects the sentiment** of every user
message (Positive / Neutral / Negative) and then asks **Llama 3** to answer
*in a way that fits that emotion* — friendly for happy customers, apologetic
and problem-solving for angry ones, plain and informative for neutral ones.

```
        USER MESSAGE
             │
             ▼
   ┌───────────────────┐
   │ Sentiment Analysis│  →  POSITIVE / NEUTRAL / NEGATIVE (+ confidence)
   └───────────────────┘
             │
             ▼
   ┌───────────────────┐
   │  Prompt Builder   │  →  injects sentiment + style rules into the prompt
   └───────────────────┘
             │
             ▼
   ┌───────────────────┐
   │     Llama 3       │  (served by Ollama)
   └───────────────────┘
             │
             ▼
     SENTIMENT-APPROPRIATE RESPONSE
```

---

## 1. Features

| Feature | Description |
|---|---|
| **Sentiment detection** | 3 classes — `POSITIVE`, `NEUTRAL`, `NEGATIVE` — with a confidence score |
| **Two backends** | `transformer` (RoBERTa sentiment model) *and* a built-in `lexicon` fallback so it runs anywhere |
| **Emotion extras** | Detects frustration intensity (e.g. ALL-CAPS, "!!!", escalation words) |
| **Sentiment-aware prompting** | The Llama 3 prompt changes per sentiment — appreciative / informative / apologetic |
| **Web chat UI** | Live chat with a sentiment badge on every bot turn |
| **Evaluation harness** | Measures sentiment accuracy, response appropriateness and a satisfaction proxy |
| **Tests** | Pytest suite for the sentiment engine and prompt builder |

---

## 2. Quick start

### 2.1 Install Llama 3 with Ollama

```bash
# macOS / Linux
brew install ollama          # or:  curl -fsSL https://ollama.com/install.sh | sh

# start the server (leave it running)
ollama serve

# in a second terminal, pull the model once
ollama pull llama3
```

### 2.2 Set up the Python app

```bash
cd "Sentiment Base Chatbot"

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env             # tweak if you like
```

> The full `transformers` + `torch` install is ~2 GB. If you want a
> lightweight run, install only `fastapi uvicorn pydantic python-dotenv httpx`
> — the chatbot will automatically use the built-in lexicon sentiment model.

### 2.3 Run

```bash
python -m app.main
# or:  uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** — the chat UI is served there.

---

## 3. Using it

Type a message in the chat box. Every bot reply carries a sentiment badge:

| User message | Detected sentiment | Bot style |
|---|---|---|
| "Your support was very helpful. Thank you!" | 😊 POSITIVE (0.98) | Warm, appreciative |
| "Can you tell me my order status?" | 😐 NEUTRAL (0.91) | Short, informative, asks for the missing detail |
| "I am very disappointed. My order has been delayed for 5 days!" | 😠 NEGATIVE (0.94) | Empathetic apology + concrete next step |

---

## 4. API

### `POST /chat`

```bash
curl -s http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"I have been waiting for my refund for 10 days. This is ridiculous!"}' | python -m json.tool
```

```json
{
  "reply": "I'm sorry about the delay with your refund. I understand how frustrating this is. Please provide your order or refund ID so I can help you check the status.",
  "sentiment": {
    "label": "NEGATIVE",
    "confidence": 0.94,
    "scores": { "NEGATIVE": 0.94, "NEUTRAL": 0.04, "POSITIVE": 0.02 },
    "backend": "lexicon",
    "signals": ["delay", "ridiculous", "!"]
  },
  "session_id": "…",
  "model": "llama3"
}
```

Other endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Web chat UI |
| `GET` | `/health` | App + Ollama status |
| `POST` | `/sentiment` | Sentiment only (no LLM call) |
| `POST` | `/reset` | Clear a session's memory |
| `GET` | `/api/demo` | The three example conversations from the assignment |

---

## 5. Evaluation

`evaluation/evaluate.py` runs the chatbot over a labelled test set
(`evaluation/test_set.json`) and reports the assignment's three criteria:

```bash
python evaluation/evaluate.py                 # sentiment accuracy + appropriateness (offline)
python evaluation/evaluate.py --with-llm      # additionally calls Llama 3 for every case
python evaluation/evaluate.py --json evaluation/results.json   # save full results
python evaluation/evaluate.py --save-baseline # snapshot this run as the baseline
python evaluation/evaluate.py --compare-baseline              # diff against the baseline
python evaluation/evaluate.py --min-accuracy 0.85 --min-appropriate 0.90   # CI gate
```

`--min-accuracy` / `--min-appropriate` turn the script into a gate: it exits
with status `1` if either metric drops below the given value, which is handy
in CI. `--save-baseline` writes `evaluation/baseline_results.json`, and
`--compare-baseline` prints a per-metric before/after diff so you can see
whether a change improved or regressed the system.

Output:

```
A. Accuracy of sentiment detection   : 0.9xx   (macro-F1, per-class report)
B. Appropriateness of responses      : 0.9xx   (rubric-based, per sentiment)
C. Impact on customer satisfaction   : normal vs sentiment-aware comparison
```

The appropriateness check uses a deterministic rubric (does a NEGATIVE reply
contain empathy + an offered action? does a POSITIVE reply acknowledge thanks?
etc.) so it is reproducible without a human grader, and `--with-llm` shows the
real generated replies for inspection.

---

## 6. Project layout

```
.
├── app/
│   ├── main.py            # FastAPI app, routes, session handling
│   ├── sentiment.py       # sentiment engine (transformer + lexicon backends)
│   ├── llm.py             # Llama 3 client (Ollama) + streaming
│   ├── prompts.py         # sentiment-aware system prompt builder
│   ├── schemas.py         # pydantic request/response models
│   └── config.py          # env-driven settings
├── static/
│   ├── index.html         # chat UI
│   ├── app.js
│   └── style.css
├── evaluation/
│   ├── evaluate.py        # evaluation harness
│   ├── test_set.json      # labelled messages
│   └── rubric.py          # response-appropriateness scoring
├── tests/
│   └── test_sentiment.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## 7. Notes for the report

* **Sentiment model choice** — the default transformer is
  `cardiffnlp/twitter-roberta-base-sentiment-latest`: it is trained on social
  text, which matches short, informal customer messages far better than a
  general-purpose review model.
* **Why a lexicon fallback exists** — graded lab machines often cannot
  download a 2 GB model. The fallback keeps the architecture honest: the same
  interface, the same three classes, just a smaller decision surface.
* **Negation handling** — the lexicon backend explicitly handles negation
  ("not good" ⇒ negative, "not bad" ⇒ positive) so accuracy does not collapse
  on the most common customer phrasing.
