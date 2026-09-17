# # from __future__ import annotations

# # from typing import Any

# # from fastapi import APIRouter, HTTPException
# # from pydantic import BaseModel, Field

# # from generation.chat import load_chatbot


# # # ================================================================
# # # ROUTER
# # # ================================================================

# # router = APIRouter()


# # # ================================================================
# # # GLOBAL CHATBOT STATE
# # # ================================================================

# # _state: dict[str, Any] = {
# #     "chatbot": None,
# # }


# # # ================================================================
# # # REQUEST MODEL
# # # ================================================================

# # class ChatRequest(BaseModel):
# #     message: str = Field(
# #         ...,
# #         min_length=1,
# #         max_length=10000,
# #         description="User message",
# #     )

# #     max_new_tokens: int = Field(
# #         default=80,
# #         ge=1,
# #         le=512,
# #         description="Maximum generated tokens",
# #     )


# # # ================================================================
# # # RESPONSE MODEL
# # # ================================================================

# # class ChatResponse(BaseModel):

# #     response: str

# #     source: str

# #     sources: list[str] = Field(
# #         default_factory=list
# #     )


# # # ================================================================
# # # LOAD CHATBOT
# # # ================================================================

# # def get_chatbot():

# #     if _state["chatbot"] is None:

# #         print()
# #         print("=" * 70)
# #         print("LOADING GENIEE MODEL FOR API")
# #         print("=" * 70)

# #         _state["chatbot"] = load_chatbot()

# #         print("Geniee model loaded.")
# #         print("=" * 70)

# #     return _state["chatbot"]


# # # ================================================================
# # # HEALTH CHECK
# # # ================================================================

# # @router.get("/health")
# # def health() -> dict[str, str]:

# #     return {
# #         "status": "ok",
# #         "service": "Geniee API",
# #     }


# # # ================================================================
# # # MODEL STATUS
# # # ================================================================

# # @router.get("/status")
# # def status() -> dict[str, Any]:

# #     chatbot_loaded = (
# #         _state["chatbot"] is not None
# #     )

# #     return {
# #         "status": "ok",
# #         "model": "Geniee 27M",
# #         "chatbot_loaded": chatbot_loaded,
# #         "checkpoint": (
# #             "checkpoints/geniee_sft_best.pt"
# #         ),
# #     }


# # # ================================================================
# # # CHAT
# # # ================================================================

# # @router.post(
# #     "/chat",
# #     response_model=ChatResponse,
# # )
# # def chat(
# #     request: ChatRequest,
# # ) -> ChatResponse:

# #     message = request.message.strip()

# #     if not message:

# #         raise HTTPException(
# #             status_code=400,
# #             detail="Message cannot be empty.",
# #         )

# #     try:

# #         chatbot = get_chatbot()

# #         response = chatbot.respond(
# #             message,
# #             max_new_tokens=request.max_new_tokens,
# #         )

# #     except (
# #         OSError,
# #         RuntimeError,
# #         ValueError,
# #     ) as exc:

# #         print()
# #         print("GENIEE API ERROR")
# #         print(exc)

# #         raise HTTPException(
# #             status_code=500,
# #             detail=str(exc),
# #         ) from exc

# #     return ChatResponse(
# #         response=response,
# #         source=chatbot.last_source,
# #         sources=chatbot.last_sources,
# #     )


# # # ================================================================
# # # OPENAI-STYLE COMPATIBILITY ENDPOINT
# # # ================================================================

# # @router.post(
# #     "/v1/chat/completions"
# # )
# # def chat_completion(
# #     request: ChatRequest,
# # ) -> dict[str, Any]:

# #     result = chat(request)

# #     return {
# #         "id": "geniee-local",
# #         "object": "chat.completion",
# #         "choices": [
# #             {
# #                 "index": 0,
# #                 "message": {
# #                     "role": "assistant",
# #                     "content": result.response,
# #                 },
# #                 "finish_reason": "stop",
# #             }
# #         ],
# #         "model": "geniee-27m",
# #     }
# from __future__ import annotations

# from typing import Any

# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel, Field

# from generation.chat import load_chatbot


# # ================================================================
# # ROUTER
# # ================================================================

# router = APIRouter()


# # ================================================================
# # GLOBAL CHATBOT STATE
# # ================================================================

# _state: dict[str, Any] = {
#     "chatbot": None,
# }


# # ================================================================
# # REQUEST MODEL
# # ================================================================

# class ChatRequest(BaseModel):
#     message: str = Field(
#         ...,
#         min_length=1,
#         max_length=10000,
#         description="User message",
#     )

#     max_new_tokens: int = Field(
#         default=80,
#         ge=1,
#         le=512,
#         description="Maximum generated tokens",
#     )

#     enable_web_search: bool = Field(
#         default=True,
#         description="Toggle live web search fallback when local context is missing",
#     )


# # ================================================================
# # RESPONSE MODEL
# # ================================================================

# class ChatResponse(BaseModel):

#     response: str

#     source: str

#     sources: list[str] = Field(
#         default_factory=list
#     )


# # ================================================================
# # LOAD CHATBOT
# # ================================================================

# def get_chatbot():

#     if _state["chatbot"] is None:

#         print()
#         print("=" * 70)
#         print("LOADING GENIEE MODEL FOR API")
#         print("=" * 70)

#         _state["chatbot"] = load_chatbot()

#         print("Geniee model loaded.")
#         print("=" * 70)

#     return _state["chatbot"]


# # ================================================================
# # HEALTH CHECK
# # ================================================================

# @router.get("/health")
# def health() -> dict[str, str]:

#     return {
#         "status": "ok",
#         "service": "Geniee API",
#     }


# # ================================================================
# # MODEL STATUS
# # ================================================================

# @router.get("/status")
# def status() -> dict[str, Any]:

#     chatbot_loaded = (
#         _state["chatbot"] is not None
#     )

#     return {
#         "status": "ok",
#         "model": "Geniee 27M",
#         "chatbot_loaded": chatbot_loaded,
#         "checkpoint": (
#             "checkpoints/geniee_sft_best.pt"
#         ),
#     }


# # ================================================================
# # CHAT
# # ================================================================

# @router.post(
#     "/chat",
#     response_model=ChatResponse,
# )
# def chat(
#     request: ChatRequest,
# ) -> ChatResponse:

#     message = request.message.strip()

#     if not message:

#         raise HTTPException(
#             status_code=400,
#             detail="Message cannot be empty.",
#         )

#     try:

#         chatbot = get_chatbot()

#         # Execute generation with optional web search override
#         try:
#             response = chatbot.respond(
#                 message,
#                 max_new_tokens=request.max_new_tokens,
#                 enable_web_search=request.enable_web_search,
#             )
#         except TypeError:
#             # Fallback if underlying chatbot.respond signature hasn't exposed kwargs yet
#             response = chatbot.respond(
#                 message,
#                 max_new_tokens=request.max_new_tokens,
#             )

#         # Retrieve source metadata safely from chatbot instance
#         source = getattr(chatbot, "last_source", "local_model")
#         sources = getattr(chatbot, "last_sources", [])

#     except (
#         OSError,
#         RuntimeError,
#         ValueError,
#     ) as exc:

#         print()
#         print("GENIEE API ERROR")
#         print(exc)

#         raise HTTPException(
#             status_code=500,
#             detail=str(exc),
#         ) from exc

#     return ChatResponse(
#         response=response,
#         source=source,
#         sources=sources,
#     )


# # ================================================================
# # OPENAI-STYLE COMPATIBILITY ENDPOINT
# # ================================================================

# @router.post(
#     "/v1/chat/completions"
# )
# def chat_completion(
#     request: ChatRequest,
# ) -> dict[str, Any]:

#     result = chat(request)

#     return {
#         "id": "geniee-local",
#         "object": "chat.completion",
#         "choices": [
#             {
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": result.response,
#                 },
#                 "finish_reason": "stop",
#             }
#         ],
#         "model": "geniee-27m",
#     }
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from generation.chat import load_chatbot


# ================================================================
# ROUTER
# ================================================================

router = APIRouter()


# ================================================================
# GLOBAL CHATBOT STATE
# ================================================================

_state: dict[str, Any] = {
    "chatbot": None,
}


# ================================================================
# REQUEST / RESPONSE MODELS
# ================================================================

class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="Role of the speaker: 'user' or 'assistant'",
    )
    content: str = Field(
        ...,
        description="Message content text",
    )


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message",
    )

    history: list[ChatMessage] = Field(
        default_factory=list,
        description="List of prior chat messages for context",
    )

    max_new_tokens: int = Field(
        default=80,
        ge=1,
        le=512,
        description="Maximum generated tokens",
    )

    enable_web_search: bool = Field(
        default=True,
        description="Toggle live web search fallback when local context is missing",
    )


class ChatResponse(BaseModel):

    response: str

    source: str

    sources: list[str] = Field(
        default_factory=list
    )


# ================================================================
# LOAD CHATBOT
# ================================================================

def get_chatbot():

    if _state["chatbot"] is None:

        print()
        print("=" * 70)
        print("LOADING GENIEE MODEL FOR API")
        print("=" * 70)

        _state["chatbot"] = load_chatbot()

        print("Geniee model loaded.")
        print("=" * 70)

    return _state["chatbot"]


# ================================================================
# HEALTH CHECK
# ================================================================

@router.get("/health")
def health() -> dict[str, str]:

    return {
        "status": "ok",
        "service": "Geniee API",
    }


# ================================================================
# MODEL STATUS
# ================================================================

@router.get("/status")
def status() -> dict[str, Any]:

    chatbot_loaded = (
        _state["chatbot"] is not None
    )

    return {
        "status": "ok",
        "model": "Geniee 27M",
        "chatbot_loaded": chatbot_loaded,
        "checkpoint": (
            "checkpoints/geniee_sft_best.pt"
        ),
    }


# ================================================================
# CHAT
# ================================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
) -> ChatResponse:

    message = request.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    try:

        chatbot = get_chatbot()

        # Format history into a dict list standard for chat templates
        formatted_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.history
        ]

        # Execute generation with graceful fallbacks for kwargs signatures
        try:
            response = chatbot.respond(
                message,
                history=formatted_history,
                max_new_tokens=request.max_new_tokens,
                enable_web_search=request.enable_web_search,
            )
        except TypeError:
            try:
                response = chatbot.respond(
                    message,
                    history=formatted_history,
                    max_new_tokens=request.max_new_tokens,
                )
            except TypeError:
                try:
                    response = chatbot.respond(
                        message,
                        max_new_tokens=request.max_new_tokens,
                        enable_web_search=request.enable_web_search,
                    )
                except TypeError:
                    response = chatbot.respond(
                        message,
                        max_new_tokens=request.max_new_tokens,
                    )

        # Retrieve source metadata safely from chatbot instance
        source = getattr(chatbot, "last_source", "local_model")
        sources = getattr(chatbot, "last_sources", [])

    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:

        print()
        print("GENIEE API ERROR")
        print(exc)

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        response=response,
        source=source,
        sources=sources,
    )


# ================================================================
# OPENAI-STYLE COMPATIBILITY ENDPOINT
# ================================================================

@router.post(
    "/v1/chat/completions"
)
def chat_completion(
    request: ChatRequest,
) -> dict[str, Any]:

    result = chat(request)

    return {
        "id": "geniee-local",
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
        "model": "geniee-27m",
    }