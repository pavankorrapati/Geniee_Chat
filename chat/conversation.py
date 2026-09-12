from __future__ import annotations

from chat.memory import ConversationMemory
from chat.prompt import DEFAULT_SYSTEM_PROMPT, build_prompt


class GenieeConversation:
    """Conversation manager that keeps the prompt format consistent with SFT."""

    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_turns: int = 6):
        self.system_prompt = system_prompt
        self.memory = ConversationMemory(max_turns=max_turns)

    def add_user(self, text: str) -> None:
        self.memory.add("user", text)

    def add_assistant(self, text: str) -> None:
        self.memory.add("assistant", text)

    def clear(self) -> None:
        self.memory.clear()

    def prompt(self) -> str:
        return build_prompt(self.system_prompt, self.memory.get_messages())
