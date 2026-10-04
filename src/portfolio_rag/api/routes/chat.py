"""Chat endpoint: a question in, a grounded answer out.

The route does four things — validate, delegate, project, return — and none of
them is retrieval, prompting or generation. It does not know what a vector
store is, which provider answers, or what the prompt says; it hands the message
to the query side and turns the result into the public shape.

Failures do not need handling here either. The query side raises
:class:`~portfolio_rag.core.errors.AppError` subclasses that the shared handler
turns into the standard envelope, which is why there is no ``try`` in this
file and no way for a provider's exception text to reach a client.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from portfolio_rag.api.dependencies import AnswerService
from portfolio_rag.api.error_handlers import (
    HTTP_413_CONTENT_TOO_LARGE,
    HTTP_422_UNPROCESSABLE_CONTENT,
)
from portfolio_rag.api.schemas.chat import ChatRequest, ChatResponse
from portfolio_rag.api.schemas.errors import ErrorResponse
from portfolio_rag.rag.conversation import ConversationTurn

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_200_OK: {
            "model": ChatResponse,
            "description": (
                "A grounded answer. `citations` lists the documents it was built "
                "from, and is empty exactly when the knowledge base does not "
                "support an answer."
            ),
        },
        status.HTTP_403_FORBIDDEN: {
            "model": ErrorResponse,
            "description": (
                "The request did not come through the gateway this service sits "
                "behind. Only returned where that gateway is configured; no "
                "retrieval or generation is attempted."
            ),
        },
        HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ErrorResponse,
            "description": "The request payload failed validation.",
        },
        HTTP_413_CONTENT_TOO_LARGE: {
            "model": ErrorResponse,
            "description": (
                "The request body exceeded 16 KiB and was refused before it was parsed."
            ),
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
            "description": (
                "A service this request depends on — the knowledge search or the "
                "generation provider — could not be reached, did not deliver a usable "
                "result, or did not finish within the request's time limit. Nothing "
                "is published in that case. Safe to retry."
            ),
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "An unexpected error. Nothing about it is disclosed.",
        },
    },
    summary="Ask the assistant a question",
    description=(
        "Answers a question from the public knowledge base. The answer is "
        "generated only from passages retrieved for this question, and every "
        "citation refers to a document that was actually retrieved.\n\n"
        "When the knowledge base does not cover the question, the response is "
        "still `200`: the answer says so and `citations` is empty. That is the "
        "honest outcome, not an error.\n\n"
        "This endpoint keeps no conversation state — `conversation_id` is "
        "echoed back and nothing more. A client may send the few turns before "
        "the question in `conversation`; they help to read a follow-up and are "
        "never treated as a source."
    ),
)
async def create_chat_completion(request: ChatRequest, answers: AnswerService) -> ChatResponse:
    result = await answers.answer(
        request.message,
        [ConversationTurn(role=turn.role, text=turn.content) for turn in request.conversation],
    )
    return ChatResponse(
        answer=result.answer,
        citations=list(result.citations),
        conversation_id=request.conversation_id,
    )
