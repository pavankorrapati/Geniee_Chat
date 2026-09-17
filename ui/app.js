const messages = document.getElementById("messages");

const input = document.getElementById("messageInput");

const sendButton = document.getElementById("sendButton");

const clearButton = document.getElementById("clearButton");

const newChatButton = document.getElementById("newChatButton");

const connectionStatus =
    document.getElementById("connectionStatus");


let isGenerating = false;


// ============================================================
// HEALTH CHECK
// ============================================================

async function checkConnection() {

    try {

        const response =
            await fetch("/api/health");

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        connectionStatus.textContent =
            "Connected to local Geniee API";

        connectionStatus.style.color =
            "#16a34a";

    } catch (error) {

        connectionStatus.textContent =
            "Geniee API offline";

        connectionStatus.style.color =
            "#dc2626";
    }
}


// ============================================================
// ADD MESSAGE
// ============================================================

function addMessage(
    role,
    text,
    source = null
) {

    const welcome =
        document.querySelector(".welcome");

    if (welcome) {
        welcome.remove();
    }


    const wrapper =
        document.createElement("div");

    wrapper.className =
        `message ${role}`;


    const avatar =
        document.createElement("div");

    avatar.className =
        `avatar ${role === "geniee"
            ? "geniee"
            : "user"}`;

    avatar.textContent =
        role === "geniee"
            ? "G"
            : "U";


    const content =
        document.createElement("div");


    const bubble =
        document.createElement("div");

    bubble.className =
        "bubble";

    bubble.textContent =
        text;


    content.appendChild(bubble);


    if (
        role === "geniee" &&
        source
    ) {

        const sourceElement =
            document.createElement("div");

        sourceElement.className =
            "source";

        sourceElement.textContent =
            `Source: ${source}`;

        content.appendChild(
            sourceElement
        );
    }


    if (role === "geniee") {

        wrapper.appendChild(
            avatar
        );

        wrapper.appendChild(
            content
        );

    } else {

        wrapper.appendChild(
            content
        );

        wrapper.appendChild(
            avatar
        );
    }


    messages.appendChild(
        wrapper
    );


    messages.scrollTop =
        messages.scrollHeight;
}


// ============================================================
// LOADING MESSAGE
// ============================================================

function addLoadingMessage() {

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "message geniee";

    wrapper.id =
        "loadingMessage";


    const avatar =
        document.createElement("div");

    avatar.className =
        "avatar geniee";

    avatar.textContent =
        "G";


    const bubble =
        document.createElement("div");

    bubble.className =
        "bubble loading";

    bubble.textContent =
        "Geniee is thinking...";


    wrapper.appendChild(
        avatar
    );

    wrapper.appendChild(
        bubble
    );


    messages.appendChild(
        wrapper
    );


    messages.scrollTop =
        messages.scrollHeight;
}


// ============================================================
// REMOVE LOADING
// ============================================================

function removeLoadingMessage() {

    const loading =
        document.getElementById(
            "loadingMessage"
        );

    if (loading) {
        loading.remove();
    }
}


// ============================================================
// SEND MESSAGE
// ============================================================

async function sendMessage(
    providedMessage = null
) {

    if (isGenerating) {
        return;
    }


    const message =
        (
            providedMessage ??
            input.value
        ).trim();


    if (!message) {
        return;
    }


    addMessage(
        "user",
        message
    );


    input.value = "";

    input.style.height = "auto";


    isGenerating = true;

    sendButton.disabled = true;


    addLoadingMessage();


    try {

        const response =
            await fetch(
                "/api/chat",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        message: message,
                        max_new_tokens: 80
                    })
                }
            );


        const data =
            await response.json();


        removeLoadingMessage();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Geniee API error"
            );
        }


        addMessage(
            "geniee",
            data.response,
            data.source
        );


    } catch (error) {

        removeLoadingMessage();


        addMessage(
            "geniee",
            `Error: ${error.message}`,
            "API error"
        );


    } finally {

        isGenerating = false;

        sendButton.disabled = false;

        input.focus();
    }
}


// ============================================================
// CLEAR CHAT
// ============================================================

function clearChat() {

    messages.innerHTML = `

        <div class="welcome">

            <div class="welcome-icon">
                G
            </div>

            <h2>
                Welcome to Geniee
            </h2>

            <p>
                Ask a question to start chatting
                with Geniee.
            </p>

        </div>
    `;
}


// ============================================================
// BUTTON EVENTS
// ============================================================

sendButton.addEventListener(
    "click",
    () => sendMessage()
);


clearButton.addEventListener(
    "click",
    clearChat
);


newChatButton.addEventListener(
    "click",
    clearChat
);


// ============================================================
// ENTER KEY
// ============================================================

input.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }
    }
);


// ============================================================
// AUTO RESIZE TEXTAREA
// ============================================================

input.addEventListener(
    "input",
    () => {

        input.style.height =
            "auto";

        input.style.height =
            `${Math.min(
                input.scrollHeight,
                150
            )}px`;
    }
);


// ============================================================
// SUGGESTION BUTTONS
// ============================================================

document.addEventListener(
    "click",
    event => {

        const button =
            event.target.closest(
                ".suggestion"
            );

        if (!button) {
            return;
        }

        const message =
            button.dataset.message;

        sendMessage(message);
    }
);


// ============================================================
// INITIALIZATION
// ============================================================

checkConnection();

input.focus();