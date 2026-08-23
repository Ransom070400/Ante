"""The real OpenAI realtime client must complete a full turn against the 0G server.

This is the contract that matters: the conversation app is an OpenAI Realtime client, so
if the SDK can connect, drive a turn and parse every event, the app works unmodified.
"""

import base64
import asyncio

import numpy as np
import pytest
import websockets
from openai import AsyncOpenAI

from ante.zerog import events
from ante.zerog.language_model import ToolCall, StreamedTurn
from ante.zerog.realtime_server import RealtimeSession


PORT = 8798


class _SpeechToText:
    async def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        return "hello robot"


class _LanguageModel:
    async def stream_turn(self, messages, tools=None):  # type: ignore[no-untyped-def]
        turn = StreamedTurn()
        for word in "Hi there friend. I run on the zero G network.".split():
            turn.text = (turn.text + " " + word).strip()
            yield "text", turn
        turn.tool_calls = [ToolCall(call_id="call_1", name="move_head", arguments='{"yaw": 10}')]
        yield "done", turn


class _TextToSpeech:
    async def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(1600, dtype=np.int16)


async def _handler(connection: websockets.asyncio.server.ServerConnection) -> None:
    session = RealtimeSession(connection, _SpeechToText(), _LanguageModel(), _TextToSpeech(), gated=False)
    await session.send(events.session_created(session.session_id))
    try:
        async for message in connection:
            await session.handle(message)
    except websockets.ConnectionClosed:
        pass


@pytest.mark.asyncio
async def test_openai_client_completes_a_turn() -> None:
    """Every event the server emits must be parseable by the client that consumes it."""
    async with websockets.serve(_handler, "127.0.0.1", PORT, max_size=None):
        client = AsyncOpenAI(
            api_key="DUMMY",
            base_url=f"http://127.0.0.1:{PORT}/v1",
            websocket_base_url=f"ws://127.0.0.1:{PORT}/v1",
        )
        observed: list[str] = []

        async def run_turn() -> None:
            async with client.realtime.connect() as connection:
                await connection.session.update(
                    session={
                        "type": "realtime",
                        "instructions": "You are Ante.",
                        "tools": [
                            {
                                "type": "function",
                                "name": "move_head",
                                "description": "move",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                )
                loud = (np.random.randn(1600) * 4000).astype(np.int16).tobytes()
                quiet = np.zeros(1600, dtype=np.int16).tobytes()
                for _ in range(4):
                    await connection.input_audio_buffer.append(audio=base64.b64encode(loud).decode())
                for _ in range(12):
                    await connection.input_audio_buffer.append(audio=base64.b64encode(quiet).decode())

                async for event in connection:
                    observed.append(event.type)
                    if event.type == "conversation.item.input_audio_transcription.completed":
                        assert event.transcript == "hello robot"
                    if event.type == "response.function_call_arguments.done":
                        assert event.name == "move_head"
                    if event.type == "response.done":
                        break

        await asyncio.wait_for(run_turn(), timeout=60)

    for required in (
        "session.created",
        "session.updated",
        "conversation.item.input_audio_transcription.completed",
        "response.created",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.function_call_arguments.done",
        "response.done",
    ):
        assert required in observed, f"client never received {required}"
