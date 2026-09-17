# from __future__ import annotations

# from chat.memory import ConversationMemory
# from chat.prompt import DEFAULT_SYSTEM_PROMPT, build_prompt


# class GenieeConversation:
#     """Conversation manager that keeps the prompt format consistent with SFT."""

#     def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_turns: int = 6):
#         self.system_prompt = system_prompt
#         self.memory = ConversationMemory(max_turns=max_turns)

#     def add_user(self, text: str) -> None:
#         self.memory.add("user", text)

#     def add_assistant(self, text: str) -> None:
#         self.memory.add("assistant", text)

#     def clear(self) -> None:
#         self.memory.clear()

#     def prompt(self) -> str:
#         return build_prompt(self.system_prompt, self.memory.get_messages())
from __future__ import annotations

from chat.memory import ConversationMemory
from chat.prompt import DEFAULT_SYSTEM_PROMPT, build_prompt
from chat.search import search_live_web


class GenieeConversation:
    """Conversation manager that keeps the prompt format consistent with SFT."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 6,
        enable_web_search: bool = True,
        search_provider: str = "duckduckgo",
    ):
        self.system_prompt = system_prompt
        self.memory = ConversationMemory(max_turns=max_turns)
        self.enable_web_search = enable_web_search
        self.search_provider = search_provider

    def add_user(self, text: str) -> None:
        self.memory.add("user", text)

    def add_assistant(self, text: str) -> None:
        self.memory.add("assistant", text)

    def clear(self) -> None:
        self.memory.clear()

    def prompt(self, local_context: str | None = None) -> str:
        """Builds the final prompt string.

        Injects local RAG context if available; otherwise falls back to real-time web search.
        """
        active_system_prompt = self.system_prompt
        context_str = None

        if local_context and local_context.strip():
            context_str = f"Local Knowledge:\n{local_context}"
        elif self.enable_web_search:
            messages = self.memory.get_messages()
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                last_query = user_messages[-1].get("content", "")
                web_snippets = search_live_web(
                    last_query, max_results=3, provider=self.search_provider
                )
                if web_snippets and web_snippets != "No live web data found.":
                    context_str = f"Live Web Data:\n{web_snippets}"

        if context_str:
            active_system_prompt = f"{self.system_prompt}\n\nContext:\n{context_str}"

        return build_prompt(active_system_prompt, self.memory.get_messages())