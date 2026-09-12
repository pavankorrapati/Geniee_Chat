from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from generation.chat import load_chatbot


router = APIRouter()
_state: dict[str, Any] = {"chatbot": None}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    max_new_tokens: int = Field(default=80, ge=1, le=512)


class ChatResponse(BaseModel):
    response: str


def get_chatbot():
    if _state["chatbot"] is None:
        _state["chatbot"] = load_chatbot()
    return _state["chatbot"]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        response = get_chatbot().respond(
            request.message,
            max_new_tokens=request.max_new_tokens,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(response=response)


@router.post("/v1/chat/completions")
def chat_completion(request: ChatRequest) -> dict[str, Any]:
    result = chat(request)
    return {
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.response},
                "finish_reason": "stop",
            }
        ],
    }
