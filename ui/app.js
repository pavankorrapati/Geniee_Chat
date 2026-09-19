const state = {
    sessionId: localStorage.getItem("geniee_session_id"),
    contextLimit: 256,
    busy: false,
    videoBusy: false,
};


// ============================================================================
// DOM ELEMENTS
// ============================================================================

const messagesEl = document.getElementById("messages");
const welcomeEl = document.getElementById("welcome");
const inputEl = document.getElementById("messageInput");
const formEl = document.getElementById("chatForm");
const sendBtn = document.getElementById("sendBtn");

const contextTokensEl =
    document.getElementById("contextTokens");

const totalTokensEl =
    document.getElementById("totalTokens");

const sourceBarEl =
    document.getElementById("sourceBar");

const historyListEl =
    document.getElementById("historyList");

const webSearchToggleEl =
    document.getElementById("webSearchToggle");

const outputTokensEl =
    document.getElementById("outputTokens");


// ============================================================================
// WEB SEARCH SETTING
// ============================================================================

const savedWebSetting =
    localStorage.getItem("geniee_web_search");

if (
    savedWebSetting !== null
    && webSearchToggleEl
) {
    webSearchToggleEl.checked =
        savedWebSetting === "true";
}

webSearchToggleEl?.addEventListener(
    "change",
    () => {
        localStorage.setItem(
            "geniee_web_search",
            String(webSearchToggleEl.checked)
        );
    }
);


// ============================================================================
// OUTPUT TOKEN SETTING
// ============================================================================

const DEFAULT_OUTPUT_TOKENS = 100;

const savedOutputTokens =
    localStorage.getItem(
        "geniee_output_tokens"
    );

if (outputTokensEl) {
    if (savedOutputTokens !== null) {
        const savedValue =
            Number(savedOutputTokens);

        const availableValues =
            Array.from(
                outputTokensEl.options
            ).map(
                option => Number(option.value)
            );

        if (
            availableValues.includes(
                savedValue
            )
        ) {
            outputTokensEl.value =
                String(savedValue);
        } else {
            outputTokensEl.value =
                String(DEFAULT_OUTPUT_TOKENS);
        }
    } else {
        outputTokensEl.value =
            String(DEFAULT_OUTPUT_TOKENS);
    }

    outputTokensEl.addEventListener(
        "change",
        () => {
            localStorage.setItem(
                "geniee_output_tokens",
                outputTokensEl.value
            );
        }
    );
}


// ============================================================================
// HTML SAFETY
// ============================================================================

function escapeHtml(text) {
    const div =
        document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}


// ============================================================================
// SESSION
// ============================================================================

function setSession(id) {
    state.sessionId = id;

    localStorage.setItem(
        "geniee_session_id",
        id
    );
}


// ============================================================================
// SCROLL
// ============================================================================

function scrollToBottom() {
    const area =
        document.getElementById(
            "chatArea"
        );

    if (!area) {
        return;
    }

    area.scrollTop =
        area.scrollHeight;
}


// ============================================================================
// WELCOME
// ============================================================================

function showWelcome(show) {
    if (!welcomeEl) {
        return;
    }

    welcomeEl.style.display =
        show ? "" : "none";
}


// ============================================================================
// CHAT MESSAGE
// ============================================================================

function addMessage(
    role,
    text,
    meta = ""
) {
    showWelcome(false);

    const row =
        document.createElement("div");

    row.className =
        `message-row ${role}`;

    const wrapper =
        document.createElement("div");

    const bubble =
        document.createElement("div");

    bubble.className = "bubble";

    bubble.innerHTML =
        escapeHtml(text)
            .replace(/\n/g, "<br>");

    wrapper.appendChild(bubble);

    if (meta) {
        const metaEl =
            document.createElement("div");

        metaEl.className =
            "message-meta";

        metaEl.textContent =
            meta;

        wrapper.appendChild(metaEl);
    }

    row.appendChild(wrapper);

    messagesEl.appendChild(row);

    scrollToBottom();
}


// ============================================================================
// TYPING INDICATOR
// ============================================================================

function addTyping() {
    const row =
        document.createElement("div");

    row.id = "typingRow";
    row.className =
        "message-row assistant";

    row.innerHTML = `
        <div class="bubble">
            <span class="typing">
                <i></i>
                <i></i>
                <i></i>
            </span>
        </div>
    `;

    messagesEl.appendChild(row);

    scrollToBottom();
}


function removeTyping() {
    document
        .getElementById("typingRow")
        ?.remove();
}


// ============================================================================
// TOKEN USAGE
// ============================================================================

function updateUsage(usage = {}) {
    const prompt =
        Number(
            usage.prompt_tokens || 0
        );

    const completion =
        Number(
            usage.completion_tokens || 0
        );

    const total =
        Number(
            usage.total_tokens || 0
        );

    state.contextLimit =
        Number(
            usage.context_limit
            || state.contextLimit
        );

    if (contextTokensEl) {
        contextTokensEl.textContent =
            `${prompt} / ${state.contextLimit}`;
    }

    if (totalTokensEl) {
        totalTokensEl.textContent =
            `${total} tokens`;
    }
}


// ============================================================================
// SOURCE DISPLAY
// ============================================================================

function showSource(data) {
    const source =
        data.source || "model";

    const sources =
        data.sources || [];

    let text =
        `Source: ${source.replaceAll(
            "_",
            " "
        )}`;

    if (sources.length) {
        text +=
            ` · ${sources
                .slice(0, 3)
                .join(", ")}`;
    }

    sourceBarEl.textContent =
        text;

    sourceBarEl.classList.remove(
        "hidden"
    );
}


// ============================================================================
// API HELPER
// ============================================================================

async function api(
    url,
    options = {}
) {
    const response =
        await fetch(
            url,
            {
                headers: {
                    "Content-Type":
                        "application/json",

                    ...(options.headers || {}),
                },

                ...options,
            }
        );

    if (!response.ok) {
        let detail =
            `HTTP ${response.status}`;

        try {
            const body =
                await response.json();

            detail =
                body.detail || detail;

        } catch (_) {
            // Ignore JSON parsing failure.
        }

        throw new Error(detail);
    }

    return response.json();
}


// ============================================================================
// HEALTH
// ============================================================================

async function loadHealth() {
    try {
        const data =
            await api("/api/health");

        const modelState =
            document.getElementById(
                "modelState"
            );

        if (modelState) {
            modelState.textContent =
                data.status === "ok"
                    ? "Online"
                    : "Unavailable";
        }

    } catch (_) {
        const modelState =
            document.getElementById(
                "modelState"
            );

        if (modelState) {
            modelState.textContent =
                "Offline";
        }
    }
}


// ============================================================================
// HISTORY
// ============================================================================

async function loadHistoryList() {
    try {
        const data =
            await api(
                "/api/sessions"
            );

        historyListEl.innerHTML =
            "";

        for (
            const session of
            (data.sessions || [])
                .slice(0, 15)
        ) {
            const button =
                document.createElement(
                    "button"
                );

            button.className =
                "history-item";

            button.textContent =
                session.title
                || "New chat";

            button.title =
                session.title
                || "New chat";

            button.dataset.sessionId =
                session.session_id;

            button.addEventListener(
                "click",
                () =>
                    openSession(
                        session.session_id
                    )
            );

            historyListEl.appendChild(
                button
            );
        }

    } catch (_) {
        historyListEl.innerHTML =
            "";
    }
}


// ============================================================================
// OPEN EXISTING SESSION
// ============================================================================

async function openSession(
    sessionId
) {
    if (state.busy) {
        return;
    }

    try {
        const data =
            await api(
                `/api/history/${encodeURIComponent(
                    sessionId
                )}`
            );

        setSession(sessionId);

        messagesEl.innerHTML =
            "";

        sourceBarEl.classList.add(
            "hidden"
        );

        const messages =
            data.messages || [];

        showWelcome(
            messages.length === 0
        );

        for (
            const message of messages
        ) {
            if (
                message.role ===
                "user"
            ) {
                addMessage(
                    "user",
                    message.content
                );

            } else if (
                message.role ===
                "assistant"
            ) {
                const usage =
                    message.usage
                    || {};

                const meta =
                    usage.total_tokens
                        ? `${message.source || "model"} · ${usage.total_tokens} tokens`
                        : "";

                addMessage(
                    "assistant",
                    message.content,
                    meta
                );
            }
        }

        const lastAssistant =
            [...messages]
                .reverse()
                .find(
                    message =>
                        message.role ===
                        "assistant"
                );

        if (
            lastAssistant?.usage
        ) {
            updateUsage(
                lastAssistant.usage
            );
        }

    } catch (error) {
        console.error(error);
    }
}


// ============================================================================
// SEND CHAT MESSAGE
// ============================================================================

async function sendMessage(text) {
    if (
        state.busy
        || !text.trim()
    ) {
        return;
    }

    state.busy = true;

    sendBtn.disabled =
        true;

    inputEl.disabled =
        true;

    addMessage(
        "user",
        text.trim()
    );

    inputEl.value = "";

    autoResize();

    addTyping();

    // ------------------------------------------------------------
    // Get selected output-token limit.
    // ------------------------------------------------------------

    const maxNewTokens =
        outputTokensEl
            ? Number(
                outputTokensEl.value
            )
            : DEFAULT_OUTPUT_TOKENS;

    try {
        const data =
            await api(
                "/api/chat",
                {
                    method: "POST",

                    body: JSON.stringify({
                        message:
                            text.trim(),

                        // IMPORTANT:
                        // This is now dynamic.
                        max_new_tokens:
                            maxNewTokens,

                        session_id:
                            state.sessionId,

                        enable_web_search:
                            webSearchToggleEl
                                ? webSearchToggleEl.checked
                                : true,
                    }),
                }
            );

        setSession(
            data.session_id
        );

        removeTyping();

        addMessage(
            "assistant",
            data.response,
            `${data.source || "model"} · ${data.usage?.completion_tokens || 0} output tokens`
        );

        updateUsage(
            data.usage
        );

        showSource(data);

        await loadHistoryList();

    } catch (error) {
        removeTyping();

        addMessage(
            "assistant",
            `I could not complete the request.\n\n${error.message}`
        );

    } finally {
        state.busy = false;

        sendBtn.disabled =
            false;

        inputEl.disabled =
            false;

        inputEl.focus();
    }
}


// ============================================================================
// NEW CHAT
// ============================================================================

async function newChat() {
    if (state.busy) {
        return;
    }

    state.sessionId =
        null;

    localStorage.removeItem(
        "geniee_session_id"
    );

    messagesEl.innerHTML =
        "";

    sourceBarEl.classList.add(
        "hidden"
    );

    showWelcome(true);

    updateUsage({
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        context_limit:
            state.contextLimit,
    });

    inputEl.focus();
}


// ============================================================================
// CLEAR CURRENT CHAT
// ============================================================================

async function clearCurrentChat() {
    if (
        !state.sessionId
        || state.busy
    ) {
        await newChat();
        return;
    }

    try {
        await api(
            `/api/history/${encodeURIComponent(
                state.sessionId
            )}`,
            {
                method: "DELETE",
            }
        );

    } catch (error) {
        console.error(error);
    }

    await newChat();

    await loadHistoryList();
}


// ============================================================================
// VIDEO GENERATION
// ============================================================================

async function generateVideo() {
    if (state.videoBusy) {
        return;
    }

    const promptEl =
        document.getElementById(
            "videoPrompt"
        );

    const durationEl =
        document.getElementById(
            "videoDuration"
        );

    const fpsEl =
        document.getElementById(
            "videoFps"
        );

    const resolutionEl =
        document.getElementById(
            "videoResolution"
        );

    const resultEl =
        document.getElementById(
            "videoResult"
        );

    const buttonEl =
        document.getElementById(
            "generateVideoBtn"
        );

    if (
        !promptEl
        || !resultEl
        || !buttonEl
    ) {
        return;
    }

    const prompt =
        promptEl.value.trim();

    if (!prompt) {
        resultEl.innerHTML =
            `<div class="video-error">
                Please enter a video prompt.
            </div>`;

        promptEl.focus();

        return;
    }

    const duration =
        Number(
            durationEl?.value || 5
        );

    const fps =
        Number(
            fpsEl?.value || 24
        );

    const resolution =
        resolutionEl?.value
        || "640x360";

    state.videoBusy = true;

    buttonEl.disabled =
        true;

    buttonEl.textContent =
        "⏳ Generating...";

    resultEl.innerHTML = `
        <div class="video-progress">
            <span class="video-spinner"></span>
            <span>
                Generating video locally...
            </span>
        </div>
    `;

    try {
        const data =
            await api(
                "/api/video",
                {
                    method: "POST",

                    body: JSON.stringify({
                        prompt,
                        duration,
                        fps,
                        resolution,
                    }),
                }
            );

        if (!data.url) {
            throw new Error(
                "Video API did not return a video URL."
            );
        }

        resultEl.innerHTML = `
            <div class="video-success">
                <video
                    class="generated-video"
                    controls
                    preload="metadata"
                >
                    <source
                        src="${escapeHtml(data.url)}"
                        type="video/mp4"
                    >
                    Your browser does not support
                    MP4 video playback.
                </video>

                <a
                    class="download-video"
                    href="${escapeHtml(data.url)}"
                    download
                >
                    ⬇ Download MP4
                </a>
            </div>
        `;

    } catch (error) {
        console.error(error);

        resultEl.innerHTML = `
            <div class="video-error">
                <strong>Video generation failed.</strong>
                <br>
                ${escapeHtml(
                    error.message
                )}
            </div>
        `;

    } finally {
        state.videoBusy = false;

        buttonEl.disabled =
            false;

        buttonEl.textContent =
            "🎬 Generate Video";
    }
}


// ============================================================================
// VIDEO ENTER KEY
// ============================================================================

const videoPromptEl =
    document.getElementById(
        "videoPrompt"
    );

videoPromptEl?.addEventListener(
    "keydown",
    event => {
        if (
            event.key === "Enter"
            && event.ctrlKey
        ) {
            event.preventDefault();

            generateVideo();
        }
    }
);


// ============================================================================
// TEXTAREA AUTO RESIZE
// ============================================================================

function autoResize() {
    inputEl.style.height =
        "auto";

    inputEl.style.height =
        Math.min(
            inputEl.scrollHeight,
            150
        ) + "px";
}


// ============================================================================
// CHAT FORM
// ============================================================================

formEl.addEventListener(
    "submit",
    event => {
        event.preventDefault();

        sendMessage(
            inputEl.value
        );
    }
);


// ============================================================================
// INPUT
// ============================================================================

inputEl.addEventListener(
    "input",
    autoResize
);

inputEl.addEventListener(
    "keydown",
    event => {
        if (
            event.key === "Enter"
            && !event.shiftKey
        ) {
            event.preventDefault();

            formEl.requestSubmit();
        }
    }
);


// ============================================================================
// BUTTON EVENTS
// ============================================================================

document
    .getElementById("newChatBtn")
    ?.addEventListener(
        "click",
        newChat
    );

document
    .getElementById("clearBtn")
    ?.addEventListener(
        "click",
        clearCurrentChat
    );

document
    .getElementById("generateVideoBtn")
    ?.addEventListener(
        "click",
        generateVideo
    );


// ============================================================================
// SUGGESTIONS
// ============================================================================

document
    .querySelectorAll(
        ".suggestions button"
    )
    .forEach(button => {
        button.addEventListener(
            "click",
            () =>
                sendMessage(
                    button.dataset.prompt
                )
        );
    });


// ============================================================================
// INITIALIZATION
// ============================================================================

(async function init() {
    await loadHealth();

    await loadHistoryList();

    if (state.sessionId) {
        await openSession(
            state.sessionId
        );
    } else {
        showWelcome(true);
    }

    inputEl.focus();
})();