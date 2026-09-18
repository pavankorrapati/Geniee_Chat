const state = {
    sessionId: localStorage.getItem("geniee_session_id"),
    contextLimit: 256,
    busy: false,
};

const messagesEl = document.getElementById("messages");
const welcomeEl = document.getElementById("welcome");
const inputEl = document.getElementById("messageInput");
const formEl = document.getElementById("chatForm");
const sendBtn = document.getElementById("sendBtn");
const contextTokensEl = document.getElementById("contextTokens");
const totalTokensEl = document.getElementById("totalTokens");
const sourceBarEl = document.getElementById("sourceBar");
const historyListEl = document.getElementById("historyList");
const webSearchToggleEl = document.getElementById("webSearchToggle");

const savedWebSetting = localStorage.getItem("geniee_web_search");
if (savedWebSetting !== null && webSearchToggleEl) {
    webSearchToggleEl.checked = savedWebSetting === "true";
}

webSearchToggleEl?.addEventListener("change", () => {
    localStorage.setItem(
        "geniee_web_search",
        String(webSearchToggleEl.checked)
    );
});

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function setSession(id) {
    state.sessionId = id;
    localStorage.setItem("geniee_session_id", id);
}

function scrollToBottom() {
    const area = document.getElementById("chatArea");
    area.scrollTop = area.scrollHeight;
}

function showWelcome(show) {
    welcomeEl.style.display = show ? "" : "none";
}

function addMessage(role, text, meta = "") {
    showWelcome(false);

    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const wrapper = document.createElement("div");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = escapeHtml(text).replace(/\n/g, "<br>");

    wrapper.appendChild(bubble);

    if (meta) {
        const metaEl = document.createElement("div");
        metaEl.className = "message-meta";
        metaEl.textContent = meta;
        wrapper.appendChild(metaEl);
    }

    row.appendChild(wrapper);
    messagesEl.appendChild(row);
    scrollToBottom();
}

function addTyping() {
    const row = document.createElement("div");
    row.id = "typingRow";
    row.className = "message-row assistant";
    row.innerHTML = `
        <div class="bubble">
            <span class="typing"><i></i><i></i><i></i></span>
        </div>`;
    messagesEl.appendChild(row);
    scrollToBottom();
}

function removeTyping() {
    document.getElementById("typingRow")?.remove();
}

function updateUsage(usage = {}) {
    const prompt = Number(usage.prompt_tokens || 0);
    const completion = Number(usage.completion_tokens || 0);
    const total = Number(usage.total_tokens || 0);
    state.contextLimit = Number(usage.context_limit || state.contextLimit);

    contextTokensEl.textContent = `${prompt} / ${state.contextLimit}`;
    totalTokensEl.textContent = `${total} tokens`;
}

function showSource(data) {
    const source = data.source || "model";
    const sources = data.sources || [];
    let text = `Source: ${source.replaceAll("_", " ")}`;

    if (sources.length) {
        text += ` · ${sources.slice(0, 3).join(", ")}`;
    }

    sourceBarEl.textContent = text;
    sourceBarEl.classList.remove("hidden");
}

async function api(url, options = {}) {
    const response = await fetch(url, {
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        ...options,
    });

    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
            const body = await response.json();
            detail = body.detail || detail;
        } catch (_) {}
        throw new Error(detail);
    }
    return response.json();
}

async function loadHealth() {
    try {
        const data = await api("/api/health");
        document.getElementById("modelState").textContent =
            data.status === "ok" ? "Online" : "Unavailable";
    } catch (_) {
        document.getElementById("modelState").textContent = "Offline";
    }
}

async function loadHistoryList() {
    try {
        const data = await api("/api/sessions");
        historyListEl.innerHTML = "";

        for (const session of (data.sessions || []).slice(0, 15)) {
            const button = document.createElement("button");
            button.className = "history-item";
            button.textContent = session.title || "New chat";
            button.title = session.title || "New chat";
            button.dataset.sessionId = session.session_id;
            button.addEventListener("click", () => openSession(session.session_id));
            historyListEl.appendChild(button);
        }
    } catch (_) {
        historyListEl.innerHTML = "";
    }
}

async function openSession(sessionId) {
    if (state.busy) return;

    try {
        const data = await api(`/api/history/${encodeURIComponent(sessionId)}`);
        setSession(sessionId);
        messagesEl.innerHTML = "";
        sourceBarEl.classList.add("hidden");

        const messages = data.messages || [];
        showWelcome(messages.length === 0);

        for (const message of messages) {
            if (message.role === "user") {
                addMessage("user", message.content);
            } else if (message.role === "assistant") {
                const usage = message.usage || {};
                const meta = usage.total_tokens
                    ? `${message.source || "model"} · ${usage.total_tokens} tokens`
                    : "";
                addMessage("assistant", message.content, meta);
            }
        }

        const lastAssistant = [...messages].reverse().find(m => m.role === "assistant");
        if (lastAssistant?.usage) updateUsage(lastAssistant.usage);
    } catch (error) {
        console.error(error);
    }
}

async function sendMessage(text) {
    if (state.busy || !text.trim()) return;

    state.busy = true;
    sendBtn.disabled = true;
    inputEl.disabled = true;

    addMessage("user", text.trim());
    inputEl.value = "";
    autoResize();
    addTyping();

    try {
        const data = await api("/api/chat", {
            method: "POST",
            body: JSON.stringify({
                message: text.trim(),
                max_new_tokens: 100,
                session_id: state.sessionId,
                enable_web_search: webSearchToggleEl
                    ? webSearchToggleEl.checked
                    : true,
            }),
        });

        setSession(data.session_id);
        removeTyping();
        addMessage(
            "assistant",
            data.response,
            `${data.source || "model"} · ${data.usage?.completion_tokens || 0} output tokens`
        );

        updateUsage(data.usage);
        showSource(data);
        await loadHistoryList();
    } catch (error) {
        removeTyping();
        addMessage("assistant", `I could not complete the request.\n\n${error.message}`);
    } finally {
        state.busy = false;
        sendBtn.disabled = false;
        inputEl.disabled = false;
        inputEl.focus();
    }
}

async function newChat() {
    if (state.busy) return;

    state.sessionId = null;
    localStorage.removeItem("geniee_session_id");
    messagesEl.innerHTML = "";
    sourceBarEl.classList.add("hidden");
    showWelcome(true);
    updateUsage({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, context_limit: state.contextLimit });
    inputEl.focus();
}

async function clearCurrentChat() {
    if (!state.sessionId || state.busy) {
        await newChat();
        return;
    }

    try {
        await api(`/api/history/${encodeURIComponent(state.sessionId)}`, {
            method: "DELETE",
        });
    } catch (error) {
        console.error(error);
    }

    await newChat();
    await loadHistoryList();
}

function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + "px";
}

formEl.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(inputEl.value);
});

inputEl.addEventListener("input", autoResize);

inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        formEl.requestSubmit();
    }
});

document.getElementById("newChatBtn").addEventListener("click", newChat);
document.getElementById("clearBtn").addEventListener("click", clearCurrentChat);

document.querySelectorAll(".suggestions button").forEach(button => {
    button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

(async function init() {
    await loadHealth();
    await loadHistoryList();

    if (state.sessionId) {
        await openSession(state.sessionId);
    } else {
        showWelcome(true);
    }

    inputEl.focus();
})();
