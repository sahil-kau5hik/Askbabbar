/* ============================================================
   AskBabbar — Conversational Chat Logic (WhatsApp Style)
   ============================================================ */

const API_BASE = "http://127.0.0.1:8000";

// ── DOM References ──────────────────────────────────────────
const queryInput         = document.getElementById("queryInput");
const queryBtn           = document.getElementById("queryBtn");
const chatMain           = document.getElementById("chatMain");
const messagesList       = document.getElementById("messagesList");
const typingIndicator    = document.getElementById("typingIndicator");
const welcomeBox         = document.getElementById("welcomeBox");
const settingsDrawer     = document.getElementById("settingsDrawer");
const topKSelect         = document.getElementById("topKSelect");
const thresholdSelect    = document.getElementById("thresholdSelect");
const searchOnlyToggle   = document.getElementById("searchOnlyToggle");
const statusDot          = document.getElementById("statusDot");
const statusText         = document.getElementById("statusText");

// ── State ───────────────────────────────────────────────────
let isLoading = false;
let messageHistory = [];

// ── Health Check ────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const data = await res.json();
      const points = data.collection_info?.points_count;
      setStatus("online", points ? `Online · ${points.toLocaleString()} chunks indexed` : "Online");
    } else {
      setStatus("offline", "API error");
    }
  } catch {
    setStatus("offline", "API offline — start run_server.py");
  }
}

function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

// ── Textarea Auto-resize ────────────────────────────────────
queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

// ── Enter to Submit (Shift+Enter for newline) ───────────────
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitQuery();
  }
});

// ── Settings Drawer Toggle ──────────────────────────────────
function toggleSettings() {
  const isHidden = settingsDrawer.style.display === "none";
  settingsDrawer.style.display = isHidden ? "block" : "none";
}

// ── Clear Chat ──────────────────────────────────────────────
function clearChat() {
  messagesList.innerHTML = "";
  messageHistory = [];
  welcomeBox.style.display = "block";
  queryInput.value = "";
  queryInput.style.height = "auto";
  queryInput.focus();
}

// ── Quick Suggestion Click ──────────────────────────────────
function sendSuggestion(text) {
  queryInput.value = text;
  submitQuery();
}

// ── Auto-scroll to bottom of chat ───────────────────────────
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatMain.scrollTop = chatMain.scrollHeight;
  });
}

// ── Format current time (e.g. 10:45 AM) ─────────────────────
function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ── HTML Escaping ───────────────────────────────────────────
function escapeHtml(s) {
  if (!s) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ── Render Markdown inside bubbles ──────────────────────────
function formatMarkdown(text) {
  if (!text) return "";

  // 1. Code blocks ```cpp ... ```
  let formatted = escapeHtml(text).replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre><code>${code.trim()}</code></pre>`;
  });

  // 2. Inline code `var`
  formatted = formatted.replace(/`([^`]+)`/g, "<code>$1</code>");

  // 3. Bold **text**
  formatted = formatted.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // 4. Line breaks
  formatted = formatted.replace(/\n/g, "<br>");

  return formatted;
}

// ── Main Submit Query ───────────────────────────────────────
async function submitQuery() {
  const query = queryInput.value.trim();
  if (!query || isLoading) return;

  // Clear input
  queryInput.value = "";
  queryInput.style.height = "auto";
  queryInput.focus();

  // Hide welcome box on first message to give maximum chat area
  if (welcomeBox) {
    welcomeBox.style.display = "none";
  }

  // 1. Append User Message Bubble
  appendUserMessage(query);

  // 2. Show Typing Indicator
  isLoading = true;
  queryBtn.disabled = true;
  typingIndicator.style.display = "flex";
  scrollToBottom();

  const top_k           = parseInt(topKSelect.value, 10);
  const threshold       = parseFloat(thresholdSelect.value);
  const searchOnly      = searchOnlyToggle.checked;
  const endpoint        = searchOnly ? "/search" : "/query";

  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        top_k,
        similarity_threshold: threshold,
        expand_adjacent: true,
      }),
      signal: AbortSignal.timeout(35000),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned HTTP ${res.status}`);
    }

    const data = await res.json();

    // 3. Hide Typing Indicator & Append Assistant Message
    typingIndicator.style.display = "none";
    isLoading = false;
    queryBtn.disabled = false;

    if (searchOnly) {
      appendAssistantMessage({
        answer: "Here are the most relevant lecture sources matching your question:",
        confidence: null,
        sources: data,
      });
    } else {
      appendAssistantMessage(data);
    }

  } catch (err) {
    typingIndicator.style.display = "none";
    isLoading = false;
    queryBtn.disabled = false;

    appendErrorMessage(err.message || "Could not reach AskBabbar server. Please verify backend is running.");
  }

  scrollToBottom();
}

// ── Append User Message ─────────────────────────────────────
function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row user-row";

  row.innerHTML = `
    <div class="message-bubble user-bubble">
      <div class="user-text">${escapeHtml(text)}</div>
      <div class="msg-time">${getCurrentTime()}</div>
    </div>
  `;

  messagesList.appendChild(row);
  messageHistory.push({ role: "user", text });
  scrollToBottom();
}

// ── Append Assistant Message ────────────────────────────────
function appendAssistantMessage(data) {
  const { answer, confidence, sources } = data;
  const row = document.createElement("div");
  row.className = "message-row assistant-row";

  const msgId = "msg-" + Date.now();
  const conf = (confidence || "balanced").toLowerCase();

  let sourcesHtml = "";
  if (sources && sources.length > 0) {
    sourcesHtml = `
      <div class="bubble-sources">
        <div class="sources-label">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
          </svg>
          Referenced Lectures (${sources.length})
        </div>
        <div class="sources-list">
          ${sources.map(src => {
            const lec = src.lecture_number ? `Lec #${src.lecture_number}` : "Lecture";
            const time = src.timestamp_range || "";
            const yt = src.youtube_url || "#";
            const title = src.title || "Love Babbar DSA Lecture";
            return `
              <div class="source-item">
                <div class="source-info">
                  <div class="source-item-title" title="${escapeHtml(title)}">${escapeHtml(title)}</div>
                  <div class="source-meta-row">
                    <span class="source-lec-tag">${escapeHtml(lec)}</span>
                    ${time ? `<span class="source-time-tag">⏱ ${escapeHtml(time)}</span>` : ""}
                  </div>
                </div>
                <a href="${yt}" target="_blank" rel="noopener noreferrer" class="source-watch-btn" title="Watch on YouTube">
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                  </svg>
                  Watch
                </a>
              </div>
            `;
          }).join("")}
        </div>
      </div>
    `;
  }

  row.innerHTML = `
    <div class="msg-avatar">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
      </svg>
    </div>
    <div class="message-bubble assistant-bubble" id="${msgId}">
      <div class="msg-top-meta">
        <span class="msg-author">Love Babbar DSA RAG</span>
        <div class="msg-actions">
          ${confidence ? `<span class="conf-pill ${conf}">${escapeHtml(conf)} confidence</span>` : ""}
          <button class="copy-icon-btn" onclick="copyMessageText('${msgId}')" title="Copy answer">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
            </svg>
          </button>
        </div>
      </div>
      <div class="bubble-content">
        ${formatMarkdown(answer || "")}
      </div>
      ${sourcesHtml}
    </div>
  `;

  messagesList.appendChild(row);
  messageHistory.push({ role: "assistant", text: answer });
  scrollToBottom();
}

// ── Append Error Message ────────────────────────────────────
function appendErrorMessage(errorText) {
  const row = document.createElement("div");
  row.className = "message-row assistant-row";

  row.innerHTML = `
    <div class="msg-avatar" style="background: rgba(239, 68, 68, 0.2); color: #ef4444;">
      ⚠️
    </div>
    <div class="message-bubble assistant-bubble" style="border-color: rgba(239, 68, 68, 0.4);">
      <div class="msg-top-meta">
        <span class="msg-author" style="color: #ef4444;">Connection Error</span>
      </div>
      <div class="bubble-content" style="color: #fca5a5;">
        ${escapeHtml(errorText)}
      </div>
    </div>
  `;

  messagesList.appendChild(row);
  scrollToBottom();
}

// ── Copy Message Text ───────────────────────────────────────
function copyMessageText(msgId) {
  const card = document.getElementById(msgId);
  if (!card) return;
  const contentEl = card.querySelector(".bubble-content");
  if (!contentEl) return;

  const rawText = contentEl.innerText;
  navigator.clipboard.writeText(rawText).then(() => {
    const btn = card.querySelector(".copy-icon-btn");
    if (btn) {
      btn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#25d366" stroke-width="2.5">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
      `;
      setTimeout(() => {
        btn.innerHTML = `
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
          </svg>
        `;
      }, 2000);
    }
  });
}

// ── Initialize ──────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30_000);
queryInput.focus();
