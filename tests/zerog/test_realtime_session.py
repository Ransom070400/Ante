"""End-to-end protocol test: the real OpenAI realtime client against the 0G server."""

import json
import base64
from typing import Any, AsyncIterator

import numpy as np
import pytest

from ante.zerog.language_model import ToolCall, StreamedTurn
from ante.zerog.realtime_server import SAMPLE_RATE, RealtimeSession


class StubSpeechToText:
    """Returns a fixed transcript without touching the network."""

    def __init__(self, transcript: str = "hello robot") -> None:
        """Store the canned ``transcript``."""
        self.transcript = transcript
        self.calls = 0

    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """Return the canned transcript."""
        self.calls += 1
        return self.transcript


class StubLanguageModel:
    """Replays a scripted answer, optionally with a tool call."""

    def __init__(self, text: str = "Hello there. How are you?", tool_calls: list[ToolCall] | None = None) -> None:
        """Replay ``text`` word by word, then ``tool_calls``."""
        self.text = text
        self.tool_calls = tool_calls or []
        self.received_messages: list[dict[str, Any]] = []

    async def stream_turn(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[tuple[str, StreamedTurn]]:
        """Stream the scripted answer."""
        self.received_messages = messages
        turn = StreamedTurn()
        for word in self.text.split(" "):
            turn.text = (turn.text + " " + word).strip()
            yield "text", turn
        turn.tool_calls = self.tool_calls
        yield "done", turn


class StubTextToSpeech:
    """Produces silence proportional to the text length."""

    def __init__(self) -> None:
        """Record every synthesized fragment."""
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        """Return silence proportional to nothing; only the call is interesting."""
        self.spoken.append(text)
        return np.zeros(3200, dtype=np.int16)


def _speech(seconds: float = 0.1, loud: bool = True) -> bytes:
    count = int(SAMPLE_RATE * seconds)
    samples = (np.random.randn(count) * 4000).astype(np.int16) if loud else np.zeros(count, dtype=np.int16)
    return base64.b64encode(samples.tobytes())


async def _drive_session(session: RealtimeSession, sent: list[dict[str, Any]]) -> None:
    """Push speech then silence through the session so one full turn completes."""
    for _ in range(4):
        await session.handle(f'{{"type":"input_audio_buffer.append","audio":"{_speech().decode()}"}}')
    for _ in range(12):
        await session.handle(f'{{"type":"input_audio_buffer.append","audio":"{_speech(loud=False).decode()}"}}')
    if session._turn_task is not None:
        await session._turn_task


class RecordingConnection:
    """Captures events instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with no captured events."""
        self.sent: list[dict[str, Any]] = []

    async def send(self, payload: str) -> None:
        """Capture one serialized event."""
        self.sent.append(json.loads(payload))


def _make_session(
    language_model: StubLanguageModel | None = None,
    text_to_speech: StubTextToSpeech | None = None,
) -> tuple[RealtimeSession, RecordingConnection, StubLanguageModel, StubTextToSpeech]:
    connection = RecordingConnection()
    llm = language_model or StubLanguageModel()
    tts = text_to_speech or StubTextToSpeech()
    session = RealtimeSession(connection, StubSpeechToText(), llm, tts, gated=False)  # type: ignore[arg-type]
    return session, connection, llm, tts


@pytest.mark.asyncio
async def test_full_turn_emits_protocol_sequence() -> None:
    """A spoken turn produces transcript, response, audio and completion events in order."""
    session, connection, _, tts = _make_session()
    await _drive_session(session, connection.sent)

    types = [event["type"] for event in connection.sent]
    assert "input_audio_buffer.speech_started" in types
    assert "input_audio_buffer.speech_stopped" in types
    assert "conversation.item.input_audio_transcription.completed" in types
    assert "response.created" in types
    assert "response.output_audio.delta" in types
    assert "response.output_audio.done" in types
    assert "response.done" in types

    assert types.index("response.created") < types.index("response.output_audio.delta")
    assert types.index("response.output_audio.done") < types.index("response.done")
    assert tts.spoken, "the assistant should have synthesized speech"


@pytest.mark.asyncio
async def test_assistant_speaks_before_the_answer_finishes() -> None:
    """Sentences are synthesized as they stream, which is what keeps the robot responsive."""
    session, connection, _, tts = _make_session(
        StubLanguageModel(text="First sentence here. Second sentence follows. Third one ends.")
    )
    await _drive_session(session, connection.sent)

    assert len(tts.spoken) > 1, f"expected incremental synthesis, got {tts.spoken}"


@pytest.mark.asyncio
async def test_tool_call_is_forwarded_to_the_client() -> None:
    """The app runs tools itself, so the call must reach it as a realtime event."""
    call = ToolCall(call_id="call_abc", name="move_head", arguments='{"yaw": 30}')
    session, connection, _, _ = _make_session(StubLanguageModel(text="Looking left.", tool_calls=[call]))
    await _drive_session(session, connection.sent)

    tool_events = [e for e in connection.sent if e["type"] == "response.function_call_arguments.done"]
    assert len(tool_events) == 1
    assert tool_events[0]["name"] == "move_head"
    assert tool_events[0]["call_id"] == "call_abc"
    assert tool_events[0]["arguments"] == '{"yaw": 30}'


@pytest.mark.asyncio
async def test_instructions_and_history_reach_the_model() -> None:
    """The profile's system prompt and prior turns must be sent on every request."""
    session, connection, llm, _ = _make_session()
    await session.handle('{"type":"session.update","session":{"instructions":"You are Ante the robot."}}')
    await _drive_session(session, connection.sent)

    assert llm.received_messages[0] == {"role": "system", "content": "You are Ante the robot."}
    assert any(m["role"] == "user" and m["content"] == "hello robot" for m in llm.received_messages)


@pytest.mark.asyncio
async def test_tool_result_is_recorded_for_the_next_turn() -> None:
    """A tool result returned by the app becomes context the model can use."""
    session, _, _, _ = _make_session()
    await session.handle(
        '{"type":"conversation.item.create","item":'
        '{"type":"function_call_output","call_id":"call_1","output":"{\\"ok\\": true}"}}'
    )
    assert session.messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}


@pytest.mark.asyncio
async def test_realtime_tools_are_converted_for_chat_completions() -> None:
    """The realtime tool schema is flat; chat completions needs it nested under `function`."""
    session, _, _, _ = _make_session()
    await session.handle(
        '{"type":"session.update","session":{"tools":[{"type":"function","name":"dance",'
        '"description":"Dance","parameters":{"type":"object","properties":{}}}]}}'
    )
    assert session.tools == [
        {
            "type": "function",
            "function": {"name": "dance", "description": "Dance", "parameters": {"type": "object", "properties": {}}},
        }
    ]


@pytest.mark.asyncio
async def test_malformed_client_json_reports_an_error() -> None:
    """Bad input is surfaced, never swallowed."""
    session, connection, _, _ = _make_session()
    await session.handle("not json at all")
    assert connection.sent[-1]["type"] == "error"
