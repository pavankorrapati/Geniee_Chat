from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    """Small bounded conversation memory for the local CLI chatbot."""
    max_turns: int = 6
    messages: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        content = content.strip()
        if content:
            self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self) -> None:
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def clear(self) -> None:
        self.messages.clear()

    def get_messages(self) -> list[dict]:
        return list(self.messages)
