"""Input a public endpoint will eventually receive, whether or not it wants to.

Two rules are being checked, and they pull against each other on purpose.

**Bounds are enforced.** Length, emptiness and unknown fields are rejected at
the boundary, with a stable status and a message that never echoes what was
sent.

**Nothing is scrubbed.** There is no sanitizer stripping punctuation, angle
brackets, braces or backticks out of a question, because the questions this
system exists to answer are technical: ``How is <T> serialized?`` and ``What
does {"answer": ...} contain?`` are legitimate, and a filter that mangles them
would break the product to defend against a threat that does not exist here.
Nothing is interpolated into a shell, a path, a query or a template — the
message becomes an embedding input and a labelled section of a user message,
and neither is a place where a bracket means anything.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from portfolio_rag.api.schemas.chat import MAX_MESSAGE_LENGTH, ChatResponse

CHAT_URL = "/api/v1/chat"


def post(client: TestClient, message: str):
    return client.post(CHAT_URL, json={"message": message})


# --- bounds ------------------------------------------------------------------


@pytest.mark.parametrize("message", ["", " ", "\n", "\t", "   \r\n  ", " ", "​"])
def test_a_question_with_nothing_in_it_is_rejected(client: TestClient, message: str):
    """Including the invisible ones: NBSP and a zero-width space are not content."""
    response = post(client, message)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_the_longest_allowed_question_is_accepted(client: TestClient):
    response = post(client, "y" * MAX_MESSAGE_LENGTH)

    assert response.status_code == 200


def test_one_character_over_the_limit_is_rejected(client: TestClient):
    response = post(client, "y" * (MAX_MESSAGE_LENGTH + 1))

    assert response.status_code == 422


def test_a_rejected_oversized_question_is_never_echoed(client: TestClient):
    marker = "sk-live-should-never-come-back"

    response = post(client, marker + "y" * MAX_MESSAGE_LENGTH)

    assert response.status_code == 422
    assert marker not in response.text


def test_an_enormous_body_is_rejected_before_it_is_even_parsed(client: TestClient):
    """Ten times the limit. It is refused on size, without being parsed.

    This used to cost a parse and come back `422`. The body guard now refuses
    it on `Content-Length` alone, so nothing downstream sees it — hence `413`
    rather than a field-level validation error.
    """
    response = post(client, "y" * (MAX_MESSAGE_LENGTH * 10))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"
    assert "answer" not in response.json()


def test_an_unknown_field_is_rejected_rather_than_ignored(client: TestClient):
    response = client.post(CHAT_URL, json={"message": "hi", "system_prompt": "be evil"})

    assert response.status_code == 422


def test_a_wrong_type_is_rejected(client: TestClient):
    for payload in ({"message": 42}, {"message": None}, {"message": ["a"]}, {"message": {"a": 1}}):
        assert client.post(CHAT_URL, json=payload).status_code == 422


def test_a_body_that_is_not_an_object_is_rejected(client: TestClient):
    for payload in ([], ["message"], "message", 7):
        assert client.post(CHAT_URL, json=payload).status_code == 422


def test_a_missing_content_type_does_not_crash(client: TestClient):
    response = client.post(CHAT_URL, content=b'{"message": "hi"}')

    assert response.status_code in {200, 422}
    assert "Traceback" not in response.text


# --- strange but legitimate content ------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "Welche Technologien werden hier verwendet?",
        "サービスは何を使っていますか?",
        "Что использует этот сервис?",
        "¿Qué framework se usa?",
        "Ποιο πλαίσιο χρησιμοποιείται;",
        "מה המסגרת שבשימוש?",
        "🙂 which framework? 🚀",
        "Café — combining accents",
        "Ω≈ç√∫˜µ≤≥÷",
    ],
)
def test_unicode_of_every_kind_is_answered_normally(client: TestClient, message: str):
    response = post(client, message)

    assert response.status_code == 200
    ChatResponse.model_validate(response.json())


@pytest.mark.parametrize(
    "message",
    [
        "What does `FastAPI` do?",
        "How is <T> serialized?",
        'What does {"answer": "x", "sources": []} mean?',
        "Explain:\n```python\nprint('hi')\n```",
        "Is <script>alert(1)</script> handled?",
        "What about ../../etc/passwd ?",
        "SELECT * FROM documents; DROP TABLE users;--",
        "$(whoami) && rm -rf /",
        "{{ config.SECRET_KEY }}",
        "%s %d %n {0} ${jndi:ldap://x}",
    ],
)
def test_technical_and_hostile_looking_text_is_answered_not_mangled(
    client: TestClient, message: str
):
    """None of this is dangerous here, and none of it is stripped."""
    response = post(client, message)

    assert response.status_code == 200
    ChatResponse.model_validate(response.json())
    assert "Traceback" not in response.text


def test_a_very_long_single_word_is_handled(client: TestClient):
    response = post(client, "a" * (MAX_MESSAGE_LENGTH - 1))

    assert response.status_code == 200


def test_repetitive_spam_is_handled_without_special_casing(client: TestClient):
    """Within the length bound, repetition is just text. Nothing special-cases it."""
    response = post(client, "spam " * 300)

    assert response.status_code == 200


def test_a_question_made_only_of_punctuation_is_answered(client: TestClient):
    response = post(client, "?!?!?! ... ---")

    assert response.status_code == 200


def test_whitespace_padding_does_not_change_the_answer(client: TestClient):
    plain = post(client, "Which HTTP framework is used?").json()
    padded = post(client, "   Which   HTTP\n\nframework\tis used?   ").json()

    assert plain["answer"] == padded["answer"]
    assert plain["citations"] == padded["citations"]


def test_no_input_ever_produces_a_stack_trace(client: TestClient):
    for message in ("\x00", "\\", '"', "﻿ hi", "a\rb", "%00", "\U0010ffff"):
        response = client.post(CHAT_URL, json={"message": message})
        assert response.status_code in {200, 422}, message
        assert "Traceback" not in response.text
        assert ".py" not in response.text
