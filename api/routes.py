from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
from chat.history import ChatHistoryStore
from generation.chat import load_chatbot
from video.generator import generate_video as create_video



PROJECT_ROOT = Path(__file__).resolve().parent.parent

router = APIRouter()
_history = ChatHistoryStore(PROJECT_ROOT / "data" / "chat_history")

# Each browser/session receives its own GenieeChat conversation state.
_sessions: dict[str, Any] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    max_new_tokens: int = Field(default=100, ge=1, le=256)
    session_id: str | None = Field(default=None, max_length=100)
    enable_web_search: bool = True


class ChatResponse(BaseModel):
    session_id: str
    response: str
    source: str
    sources: list[str]
    usage: dict[str, int]
    web_search_enabled: bool
class VideoResolution(str, Enum):
    RES_360P = "640x360"
    RES_480P = "854x480"
    RES_720P = "1280x720"    
# class VideoRequest(BaseModel):
#     prompt: str = Field(min_length=1, max_length=2000)
#     duration: int | None = Field(default=None, ge=2, le=20)
#     fps: int | None = Field(default=None, ge=12, le=30)
#     resolution: str | None = Field(default=None, pattern=r"^(640x360|854x480|1280x720)$")
class VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    duration: int | None = Field(default=None, ge=2, le=20)
    fps: int | None = Field(default=None, ge=12, le=30)
    resolution: VideoResolution | None = Field(default=None)

def _get_chatbot_and_id(
    session_id: str | None,
) -> tuple[Any, str]:
    if session_id and session_id in _sessions:
        return _sessions[session_id], session_id

    resolved_id = session_id or str(uuid4())

    chatbot = load_chatbot()
    saved_messages = _history.load(resolved_id)
    chatbot.restore_messages(saved_messages)

    _sessions[resolved_id] = chatbot
    return chatbot, resolved_id


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status")
def status() -> dict[str, Any]:
    loaded = next(iter(_sessions.values()), None)

    return {
        "status": "ok",
        "model": "Geniee 27M",
        "checkpoint": "geniee_sft_best.pt",
        "model_loaded": loaded is not None,
        "sessions_in_memory": len(_sessions),
        "max_context_tokens": (
            loaded.context_limit if loaded is not None else 256
        ),
        "features": {
            "history": True,
            "token_usage": True,
            "local_retrieval": True,
            "web_retrieval": True,
            "python_debugger": True,
        },
    }


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        chatbot, session_id = _get_chatbot_and_id(
            request.session_id
        )

        response = chatbot.respond(
            request.message,
            max_new_tokens=request.max_new_tokens,
            enable_web_search=request.enable_web_search,
        )

        usage = dict(chatbot.last_usage)

        _history.append(
            session_id=session_id,
            user_message=request.message,
            assistant_message=response,
            source=chatbot.last_source,
            usage=usage,
        )

        return ChatResponse(
            session_id=session_id,
            response=response,
            source=chatbot.last_source,
            sources=list(chatbot.last_sources),
            usage=usage,
            web_search_enabled=request.enable_web_search,
        )

    except (
        OSError,
        RuntimeError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@router.get("/history/{session_id}")
def history(session_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "messages": _history.messages(session_id),
    }


@router.delete("/history/{session_id}")
def clear_history(session_id: str) -> dict[str, Any]:
    _history.clear(session_id)
    _sessions.pop(session_id, None)

    return {
        "status": "cleared",
        "session_id": session_id,
    }


@router.get("/sessions")
def sessions() -> dict[str, Any]:
    return {
        "sessions": _history.sessions(),
    }


@router.post("/v1/chat/completions")
def chat_completion(
    request: ChatRequest,
) -> dict[str, Any]:
    result = chat(request)

    return {
        "id": f"chatcmpl-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.response,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": result.usage,
        "session_id": result.session_id,
        "source": result.source,
        "sources": result.sources,
        "web_search_enabled": result.web_search_enabled,
    }
@router.post("/video")
def generate_video(request: VideoRequest):

    try:
        result = create_video(
            prompt=request.prompt,
            duration=request.duration,
            fps=request.fps,
            resolution=request.resolution,
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video generation failed: {exc}",
        )
