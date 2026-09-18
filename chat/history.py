from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ChatHistoryStore:
    """Small file-backed history store; no database dependency is required."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(
            ch for ch in session_id if ch.isalnum() or ch in "-_"
        )[:100]
        if not safe:
            raise ValueError("Invalid session_id")
        return self.root / f"{safe}.json"

    def _read(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        if not path.exists():
            return {
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "messages": [],
            }
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError
            data.setdefault("messages", [])
            return data
        except (OSError, json.JSONDecodeError, ValueError):
            return {
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "messages": [],
            }

    def _write(self, session_id: str, data: dict[str, Any]) -> None:
        path = self._path(session_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    def append(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        source: str = "model",
        usage: dict[str, int] | None = None,
    ) -> None:
        data = self._read(session_id)
        data["messages"].append(
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        data["messages"].append(
            {
                "role": "assistant",
                "content": assistant_message,
                "source": source,
                "usage": usage or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write(session_id, data)

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        return self._read(session_id).get("messages", [])

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Return only role/content pairs suitable for GenieeConversation."""
        result = []
        for message in self.messages(session_id):
            role = message.get("role")
            content = message.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                result.append({"role": role, "content": content})
        return result

    def clear(self, session_id: str) -> None:
        path = self._path(session_id)
        if path.exists():
            path.unlink()

    def sessions(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                messages = data.get("messages", [])
                first_user = next(
                    (
                        m.get("content", "")
                        for m in messages
                        if m.get("role") == "user"
                    ),
                    "New chat",
                )
                result.append(
                    {
                        "session_id": data.get("session_id", path.stem),
                        "title": first_user[:60] or "New chat",
                        "message_count": len(messages),
                        "updated_at": (
                            messages[-1].get("timestamp")
                            if messages else data.get("created_at")
                        ),
                    }
                )
            except (OSError, json.JSONDecodeError):
                continue
        return result