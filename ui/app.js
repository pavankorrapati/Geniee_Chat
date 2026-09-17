// const messages = document.getElementById("messages");

// const input = document.getElementById("messageInput");

// const sendButton = document.getElementById("sendButton");

// const clearButton = document.getElementById("clearButton");

// const newChatButton = document.getElementById("newChatButton");

// const connectionStatus =
//     document.getElementById("connectionStatus");


// let isGenerating = false;


// // ============================================================
// // HEALTH CHECK
// // ============================================================

// async function checkConnection() {

//     try {

//         const response =
//             await fetch("/api/health");

//         if (!response.ok) {
//             throw new Error("API unavailable");
//         }

//         connectionStatus.textContent =
//             "Connected to local Geniee API";

//         connectionStatus.style.color =
//             "#16a34a";

//     } catch (error) {

//         connectionStatus.textContent =
//             "Geniee API offline";

//         connectionStatus.style.color =
//             "#dc2626";
//     }
// }


// // ============================================================
// // ADD MESSAGE
// // ============================================================




// // Locate addMessage function and update avatar creation:
// function addMessage(role, text, source = null) {
//     const welcome = document.querySelector(".welcome");
//     if (welcome) welcome.remove();

//     const wrapper = document.createElement("div");
//     wrapper.className = `message ${role}`;

//     const avatar = document.createElement("div");
//     avatar.className = `avatar ${role === "geniee" ? "geniee" : "user"}`;
    
//     // Set unique avatars & hover titles for Geniee and Flashman
//     avatar.textContent = role === "geniee" ? "🧞" : "⚡";
//     avatar.title = role === "geniee" ? "Geniee" : "Flashman";

//     const content = document.createElement("div");
//     const bubble = document.createElement("div");
//     bubble.className = "bubble";
//     bubble.textContent = text;

//     content.appendChild(bubble);

//     if (role === "geniee" && source) {
//         const sourceElement = document.createElement("div");
//         sourceElement.className = "source";
//         sourceElement.textContent = `Source: ${source}`;
//         content.appendChild(sourceElement);
//     }

//     if (role === "geniee") {
//         wrapper.appendChild(avatar);
//         wrapper.appendChild(content);
//     } else {
//         wrapper.appendChild(content);
//         wrapper.appendChild(avatar);
//     }

//     messages.appendChild(wrapper);
//     messages.scrollTop = messages.scrollHeight;
// }

// // Locate addLoadingMessage function and update avatar:
// // function addLoadingMessage() {
// //     const wrapper = document.createElement("div");
// //     wrapper.className = "message geniee";
// //     wrapper.id = "loadingMessage";

// //     const avatar = document.createElement("div");
// //     avatar.className = "avatar geniee";
// //     avatar.textContent = "🧞";
// //     avatar.title = "Geniee";

// //     const bubble = document.createElement("div");
// //     bubble.className = "bubble loading";
// //     bubble.textContent = "Geniee is thinking...";

// //     wrapper.appendChild(avatar);
// //     wrapper.appendChild(bubble);

// //     messages.appendChild(wrapper);
// //     messages.scrollTop = messages.scrollHeight;
// // }
// // ============================================================
// // ADD LOADING MESSAGE
// // ============================================================

// function addLoadingMessage() {
//     const loadingHTML = `
//       <div class="message geniee" id="loadingMessage">
//         <div class="avatar geniee" title="Geniee">🧞</div>
//         <div class="bubble loading">
//           Geniee is thinking...
//         </div>
//       </div>
//     `;

//     messages.insertAdjacentHTML("beforeend", loadingHTML);
//     messages.scrollTop = messages.scrollHeight;
// }


// // ============================================================
// // REMOVE LOADING
// // ============================================================

// function removeLoadingMessage() {

//     const loading =
//         document.getElementById(
//             "loadingMessage"
//         );

//     if (loading) {
//         loading.remove();
//     }
// }


// // ============================================================
// // SEND MESSAGE
// // ============================================================

// async function sendMessage(
//     providedMessage = null
// ) {

//     if (isGenerating) {
//         return;
//     }


//     const message =
//         (
//             providedMessage ??
//             input.value
//         ).trim();


//     if (!message) {
//         return;
//     }


//     addMessage(
//         "user",
//         message
//     );


//     input.value = "";

//     input.style.height = "auto";


//     isGenerating = true;

//     sendButton.disabled = true;


//     addLoadingMessage();


//     try {

//         const response =
//             await fetch(
//                 "/api/chat",
//                 {
//                     method: "POST",

//                     headers: {
//                         "Content-Type":
//                             "application/json"
//                     },

//                     body: JSON.stringify({
//                         message: message,
//                         max_new_tokens: 80
//                     })
//                 }
//             );


//         const data =
//             await response.json();


//         removeLoadingMessage();


//         if (!response.ok) {

//             throw new Error(
//                 data.detail ||
//                 "Geniee API error"
//             );
//         }


//         addMessage(
//             "geniee",
//             data.response,
//             data.source
//         );


//     } catch (error) {

//         removeLoadingMessage();


//         addMessage(
//             "geniee",
//             `Error: ${error.message}`,
//             "API error"
//         );


//     } finally {

//         isGenerating = false;

//         sendButton.disabled = false;

//         input.focus();
//     }
// }


// // ============================================================
// // CLEAR CHAT
// // ============================================================

// function clearChat() {

//     messages.innerHTML = `

//         <div class="welcome">

//             <div class="welcome-icon">
//                 G
//             </div>

//             <h2>
//                 Welcome to Geniee
//             </h2>

//             <p>
//                 Ask a question to start chatting
//                 with Geniee.
//             </p>

//         </div>
//     `;
// }


// // ============================================================
// // BUTTON EVENTS
// // ============================================================

// sendButton.addEventListener(
//     "click",
//     () => sendMessage()
// );


// clearButton.addEventListener(
//     "click",
//     clearChat
// );


// newChatButton.addEventListener(
//     "click",
//     clearChat
// );


// // ============================================================
// // ENTER KEY
// // ============================================================

// input.addEventListener(
//     "keydown",
//     event => {

//         if (
//             event.key === "Enter" &&
//             !event.shiftKey
//         ) {

//             event.preventDefault();

//             sendMessage();
//         }
//     }
// );


// // ============================================================
// // AUTO RESIZE TEXTAREA
// // ============================================================

// input.addEventListener(
//     "input",
//     () => {

//         input.style.height =
//             "auto";

//         input.style.height =
//             `${Math.min(
//                 input.scrollHeight,
//                 150
//             )}px`;
//     }
// );


// // ============================================================
// // SUGGESTION BUTTONS
// // ============================================================

// document.addEventListener(
//     "click",
//     event => {

//         const button =
//             event.target.closest(
//                 ".suggestion"
//             );

//         if (!button) {
//             return;
//         }

//         const message =
//             button.dataset.message;

//         sendMessage(message);
//     }
// );


// // ============================================================
// // INITIALIZATION
// // ============================================================

// checkConnection();

// input.focus();
const messages = document.getElementById("messages");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const clearButton = document.getElementById("clearButton");
const newChatButton = document.getElementById("newChatButton");
const connectionStatus = document.getElementById("connectionStatus");

let isGenerating = false;
let chatHistory = []; // Stores the last 20 messages for context


// ============================================================
// HEALTH CHECK
// ============================================================

async function checkConnection() {
    try {
        const response = await fetch("/api/health");

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        connectionStatus.textContent = "Connected to local Geniee API";
        connectionStatus.style.color = "#16a34a";
    } catch (error) {
        connectionStatus.textContent = "Geniee API offline";
        connectionStatus.style.color = "#dc2626";
    }
}


// ============================================================
// ADD MESSAGE
// ============================================================

function addMessage(role, text, source = null) {
    const welcome = document.querySelector(".welcome");
    if (welcome) welcome.remove();

    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${role === "geniee" ? "geniee" : "user"}`;
    avatar.textContent = role === "geniee" ? "🧞" : "⚡";
    avatar.title = role === "geniee" ? "Geniee" : "Flashman";

    const content = document.createElement("div");
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    content.appendChild(bubble);

    if (role === "geniee" && source) {
        const sourceElement = document.createElement("div");
        sourceElement.className = "source";
        sourceElement.textContent = `Source: ${source}`;
        content.appendChild(sourceElement);
    }

    if (role === "geniee") {
        wrapper.appendChild(avatar);
        wrapper.appendChild(content);
    } else {
        wrapper.appendChild(content);
        wrapper.appendChild(avatar);
    }

    messages.appendChild(wrapper);
    messages.scrollTop = messages.scrollHeight;
}


// ============================================================
// ADD LOADING MESSAGE
// ============================================================

function addLoadingMessage() {
    const loadingHTML = `
      <div class="message geniee" id="loadingMessage">
        <div class="avatar geniee" title="Geniee">🧞</div>
        <div class="bubble loading">
          Geniee is thinking...
        </div>
      </div>
    `;

    messages.insertAdjacentHTML("beforeend", loadingHTML);
    messages.scrollTop = messages.scrollHeight;
}


// ============================================================
// REMOVE LOADING
// ============================================================

function removeLoadingMessage() {
    const loading = document.getElementById("loadingMessage");

    if (loading) {
        loading.remove();
    }
}


// ============================================================
// SEND MESSAGE
// ============================================================

async function sendMessage(providedMessage = null) {
    if (isGenerating) {
        return;
    }

    const message = (providedMessage ?? input.value).trim();

    if (!message) {
        return;
    }

    addMessage("user", message);

    // Append user query to chat history array
    chatHistory.push({ role: "user", content: message });
    if (chatHistory.length > 20) {
        chatHistory = chatHistory.slice(-20);
    }

    input.value = "";
    input.style.height = "auto";

    isGenerating = true;
    sendButton.disabled = true;

    addLoadingMessage();

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory, // Sending context history payload
                max_new_tokens: 80
            })
        });

        const data = await response.json();

        removeLoadingMessage();

        if (!response.ok) {
            throw new Error(data.detail || "Geniee API error");
        }

        addMessage("geniee", data.response, data.source);

        // Append assistant response to chat history array
        chatHistory.push({ role: "assistant", content: data.response });
        if (chatHistory.length > 20) {
            chatHistory = chatHistory.slice(-20);
        }

    } catch (error) {
        removeLoadingMessage();

        // Roll back prompt from memory on failure
        chatHistory.pop();

        addMessage("geniee", `Error: ${error.message}`, "API error");
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
    chatHistory = []; // Reset stored conversation memory

    messages.innerHTML = `
        <div class="welcome">
            <div class="welcome-icon">
                G
            </div>
            <h2>
                Welcome to Geniee
            </h2>
            <p>
                Ask a question to start chatting with Geniee.
            </p>
        </div>
    `;
}


// ============================================================
// BUTTON EVENTS
// ============================================================

sendButton.addEventListener("click", () => sendMessage());
clearButton.addEventListener("click", clearChat);
newChatButton.addEventListener("click", clearChat);


// ============================================================
// ENTER KEY
// ============================================================

input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});


// ============================================================
// AUTO RESIZE TEXTAREA
// ============================================================

input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});


// ============================================================
// SUGGESTION BUTTONS
// ============================================================

document.addEventListener("click", event => {
    const button = event.target.closest(".suggestion");

    if (!button) {
        return;
    }

    const message = button.dataset.message;
    sendMessage(message);
});


// ============================================================
// INITIALIZATION
// ============================================================

checkConnection();
input.focus();