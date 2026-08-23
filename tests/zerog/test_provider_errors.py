"""A provider refusal must reach the user, not read as silence."""

import httpx
import numpy as np
import pytest

from ante.zerog.errors import ZeroGRequestError
from ante.zerog.language_model import _raise_for_error_body
from ante.zerog.speech_to_text import ZeroGSpeechToText, raise_for_provider_error


UNFUNDED_BODY = {
    "error": (
        "validate request: insufficient balance: your locked balance is 0.000000 0G, "
        "but the required minimum is 1.000000 0G"
    )
}


def test_clean_payload_passes_through() -> None:
    """A normal transcription result is not mistaken for an error."""
    raise_for_provider_error({"text": "hello robot"})


def test_error_in_a_successful_body_is_raised() -> None:
    """Providers report refusals inside HTTP 200, where raise_for_status cannot see them."""
    with pytest.raises(ZeroGRequestError, match="insufficient balance"):
        raise_for_provider_error(UNFUNDED_BODY)


def test_structured_error_object_is_unwrapped() -> None:
    """Some providers nest the message inside an error object."""
    with pytest.raises(ZeroGRequestError, match="model overloaded"):
        raise_for_provider_error({"error": {"message": "model overloaded"}})


@pytest.mark.asyncio
async def test_unfunded_transcription_raises_rather_than_returning_empty() -> None:
    """An unfunded sub-account must not look like an utterance nobody could hear."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=UNFUNDED_BODY)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    speech_to_text = ZeroGSpeechToText("http://0g.local/v1", "openai/whisper-large-v3", client)

    with pytest.raises(ZeroGRequestError, match="insufficient balance"):
        await speech_to_text.transcribe(np.zeros(1600, dtype=np.int16), 16000)


@pytest.mark.asyncio
async def test_successful_transcription_returns_text() -> None:
    """The happy path still works."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "  what is the capital of France?  "})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    speech_to_text = ZeroGSpeechToText("http://0g.local/v1", "openai/whisper-large-v3", client)

    assert await speech_to_text.transcribe(np.zeros(1600, dtype=np.int16), 16000) == ("what is the capital of France?")


def test_chat_error_body_is_raised() -> None:
    """A refusal returned in place of an event stream must not be parsed as content."""
    with pytest.raises(ZeroGRequestError, match="insufficient balance"):
        _raise_for_error_body('{"error": "insufficient balance"}')


def test_unreadable_chat_body_is_reported() -> None:
    """An HTML error page must produce a clear failure, not a JSON traceback."""
    with pytest.raises(ZeroGRequestError, match="unreadable"):
        _raise_for_error_body("<html>502 Bad Gateway</html>")
