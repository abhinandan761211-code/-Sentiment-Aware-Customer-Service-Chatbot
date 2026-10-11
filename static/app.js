/* ─────────────────────────────────────────────────────────────
   Sentiment-Aware Chatbot — front-end
   ───────────────────────────────────────────────────────────── */

const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const resetBtn = document.getElementById("reset-btn");

let sessionId = null;

const SENTIMENT_META = {
  POSITIVE: { emoji: "😊", label: "Positive" },
  NEUTRAL:  { emoji: "😐", label: "Neutral" },
  NEGATIVE: { emoji: "😠", label: "Negative" },
};

const ESCAPE = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESCAPE[c]);

/* ── health/status ─────────────────────────────────────────── */

async function loadStatus() {
  const backendEl = document.getElementById("s-backend");
  const ollamaEl = document.getElementById("s-ollama");
  const modelEl = document.getElementById("s-model");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    backendEl.textContent = data.sentiment_backend;
    backendEl.className = "ok";

    const ok = data.ollama === "ok";
    ollamaEl.textContent = ok ? "connected" : data.ollama.replace("-", " ");
    ollamaEl.className = ok ? "ok" : "bad";

    modelEl.textContent = data.model;
  } catch {
    backendEl.textContent = "offline";
    backendEl.className = "bad";
    ollamaEl.textContent = "unknown";
    ollamaEl.className = "bad";
  }
}

/* ── rendering ─────────────────────────────────────────────── */

function addUserMessage(text) {
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  wrap.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  messagesEl.appendChild(wrap);
  scrollDown();
}

function sentimentBadge(sentiment) {
  const meta = SENTIMENT_META[sentiment.label] || SENTIMENT_META.NEUTRAL;
  const pct = Math.round(sentiment.confidence * 100);

  const intensityBar =
    sentiment.intensity > 0
      ? `<span class="intensity" title="Emotional intensity">
           <span class="track"><span class="fill" style="width:${Math.round(
             sentiment.intensity * 100
           )}%"></span></span>
         </span>`
      : "";

  const signals = sentiment.signals && sentiment.signals.length
    ? `<span class="signals">cues: <em>${esc(sentiment.signals.join(", "))}</em></span>`
    : "";

  return `
    <div class="meta">
      <span class="badge ${sentiment.label}">${meta.emoji} ${meta.label}</span>
      <span class="confidence">${pct}% confidence</span>
      ${intensityBar}
      ${signals}
    </div>`;
}

function addBotMessage(text, sentiment) {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.innerHTML = `<div class="bubble">${esc(text)}</div>${
    sentiment ? sentimentBadge(sentiment) : ""
  }`;
  messagesEl.appendChild(wrap);
  scrollDown();
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.dataset.typing = "1";
  wrap.innerHTML = `<div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>`;
  messagesEl.appendChild(wrap);
  scrollDown();
  return wrap;
}

function addError(text) {
  const el = document.createElement("div");
  el.className = "error";
  el.textContent = text;
  messagesEl.appendChild(el);
  scrollDown();
}

function scrollDown() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ── send ──────────────────────────────────────────────────── */

async function sendMessage(text) {
  addUserMessage(text);
  const typing = addTyping();

  sendBtn.disabled = true;
  inputEl.disabled = true;

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    const data = await res.json();
    typing.remove();

    if (!res.ok) {
      addError(
        data.detail ||
          "The chatbot could not answer. Make sure Ollama is running (`ollama serve`)."
      );
      return;
    }

    sessionId = data.session_id;
    addBotMessage(data.reply, data.sentiment);
  } catch (err) {
    typing.remove();
    addError("Network error — is the server running?");
  } finally {
    sendBtn.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
  }
}

/* ── events ────────────────────────────────────────────────── */

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  sendMessage(text);
});

document.querySelectorAll(".examples button").forEach((btn) => {
  btn.addEventListener("click", () => sendMessage(btn.dataset.msg));
});

resetBtn.addEventListener("click", async () => {
  if (sessionId) {
    try {
      await fetch("/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch {
      /* ignore — the UI is cleared either way */
    }
  }
  sessionId = null;
  messagesEl.innerHTML = `
    <div class="msg bot">
      <div class="bubble">Conversation cleared. What can I help you with?</div>
    </div>`;
});

loadStatus();
inputEl.focus();
