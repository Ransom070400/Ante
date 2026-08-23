"""Every server event must parse as the client's own typed model."""

import json

import pytest
from pydantic import TypeAdapter
from openai.types.realtime.realtime_server_event import RealtimeServerEvent

from ante.zerog import events


_adapter: TypeAdapter[RealtimeServerEvent] = TypeAdapter(RealtimeServerEvent)

ALL_EVENTS = [
    events.session_created("sess_1"),
    events.session_updated("sess_1"),
    events.speech_started("item_1", 0),
    events.speech_stopped("item_1", 1200),
    events.transcription_completed("item_1", "what is the weather"),
    events.response_created("resp_1"),
    events.response_done("resp_1"),
    events.response_done("resp_1", status="cancelled"),
    events.audio_delta("resp_1", "item_1", "AAAA"),
    events.audio_done("resp_1", "item_1"),
    events.audio_transcript_done("resp_1", "item_1", "it is sunny"),
    events.function_call_done("resp_1", "item_1", "call_1", "move_head", '{"yaw": 20}'),
    events.error("something broke"),
]


@pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda event: str(event["type"]))
def test_event_parses_as_client_model(event: dict[str, object]) -> None:
    """A schema change in the OpenAI client must fail here, not mid-conversation."""
    parsed = _adapter.validate_python(event)
    assert parsed.type == event["type"]


@pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda event: str(event["type"]))
def test_event_is_json_serializable(event: dict[str, object]) -> None:
    """Events travel over a websocket as JSON text."""
    assert json.loads(json.dumps(event))["type"] == event["type"]


def test_event_ids_are_unique() -> None:
    """Duplicate event ids confuse client-side deduplication."""
    ids = [events.response_created("resp_1")["event_id"] for _ in range(50)]
    assert len(set(ids)) == len(ids)
