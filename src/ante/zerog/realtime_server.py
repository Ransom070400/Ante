"""An OpenAI Realtime protocol server backed by the 0G Compute Network.

The conversation app is an OpenAI Realtime client, so speaking that protocol lets the
entire app — tools, movement manager, profiles, Gradio UI — run unmodified against 0G.
The realtime API is speech-to-speech over one socket; 0G exposes plain request/response
models, so this server reassembles the three stages behind that interface:

    mic ─> VAD ─> 0G Whisper ─> 0G chat model ─> local Piper ─> speaker
                                      └─> tool calls ─> the app's own tool manager
"""

import os
import re
import json
import base64
import asyncio
import logging
from typing import Any
from pathlib import Path

import httpx
import numpy as np
import websockets
from numpy.typing import NDArray
from websockets.asyncio.server import ServerConnection

from ante.zerog import events
from ante.zerog.gateway import ZeroGGateway
from ante.zerog.providers import LANGUAGE_MODEL, SPEECH_TO_TEXT
from ante.zerog.antenna_gate import AntennaGate
from ante.zerog.language_model import ToolCall, ZeroGLanguageModel
from ante.zerog.speech_to_text import ZeroGSpeechToText
from ante.zerog.text_to_speech import LocalTextToSpeech
from ante.zerog.voice_activity import TurnEvent, VoiceActivityDetector


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Matches the client's expectation of ~20ms frames; also the granularity of barge-in.
_AUDIO_CHUNK_SAMPLES = 1600
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")
# Below this, a fragment is not worth a synthesis round-trip on its own.
_MIN_SPEAKABLE_CHARS = 12
PIPER_VOICE_ENV = "ANTE_PIPER_VOICE"


def _configured_piper_voice() -> Path | None:
    """Return the Piper voice file from the environment, if one is configured."""
    voice = (os.getenv(PIPER_VOICE_ENV) or "").strip()
    if not voice:
        return None

    voice_path = Path(voice).expanduser()
    if not voice_path.is_file():
        logger.warning("%s points at %s, which does not exist; falling back", PIPER_VOICE_ENV, voice_path)
        return None
    return voice_path


class RealtimeSession:
    """One client connection: conversation state plus the turn pipeline."""

    def __init__(
        self,
        connection: ServerConnection,
        speech_to_text: ZeroGSpeechToText,
        language_model: ZeroGLanguageModel,
        text_to_speech: LocalTextToSpeech,
        gated: bool,
    ) -> None:
        """Bind one websocket connection to the three 0G-backed pipeline stages."""
        self.connection = connection
        self.speech_to_text = speech_to_text
        self.language_model = language_model
        self.text_to_speech = text_to_speech

        self.session_id = f"sess_{id(self):x}"
        self.instructions = "You are a helpful robot assistant."
        self.tools: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

        self.detector = VoiceActivityDetector(SAMPLE_RATE, gated=gated)
        self._utterance: list[NDArray[np.int16]] = []
        self._turn_task: asyncio.Task[None] | None = None
        self._response_counter = 0

    async def send(self, event: dict[str, Any]) -> None:
        """Send one server event, tolerating a client that has already gone away."""
        try:
            await self.connection.send(json.dumps(event))
        except websockets.ConnectionClosed:
            logger.debug("Dropped %s: client disconnected", event.get("type"))

    async def handle(self, message: str | bytes) -> None:
        """Dispatch one client event."""
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            await self.send(events.error("Malformed JSON from client", code="invalid_request_error"))
            return

        event_type = event.get("type")
        if event_type == "session.update":
            self._apply_session_update(event.get("session") or {})
            await self.send(events.session_updated(self.session_id))
        elif event_type == "input_audio_buffer.append":
            await self._append_audio(event.get("audio") or "")
        elif event_type == "conversation.item.create":
            self._append_conversation_item(event.get("item") or {})
        elif event_type == "response.create":
            self._start_turn_task(transcript=None)
        elif event_type == "input_audio_buffer.clear":
            self._utterance.clear()
            self.detector.reset()
        else:
            logger.debug("Ignoring unhandled client event: %s", event_type)

    def _apply_session_update(self, session: dict[str, Any]) -> None:
        """Adopt instructions and tools; audio format is fixed at 16 kHz PCM16 mono."""
        instructions = session.get("instructions")
        if instructions:
            self.instructions = instructions

        tools = session.get("tools")
        if tools is not None:
            self.tools = [_to_chat_completions_tool(tool) for tool in tools]
            logger.info("Session configured with %d tools", len(self.tools))

    def _append_conversation_item(self, item: dict[str, Any]) -> None:
        """Record a client-supplied item: tool results, injected text, or images."""
        item_type = item.get("type")

        if item_type == "function_call_output":
            self.messages.append(
                {"role": "tool", "tool_call_id": item.get("call_id", ""), "content": str(item.get("output", ""))}
            )
            return

        text = _extract_text(item)
        if text:
            self.messages.append({"role": item.get("role", "user"), "content": text})

    async def _append_audio(self, encoded_audio: str) -> None:
        """Feed one microphone frame through turn detection."""
        try:
            samples = np.frombuffer(base64.b64decode(encoded_audio), dtype=np.int16)
        except (ValueError, TypeError) as exc:
            logger.warning("Discarding undecodable audio frame: %s", exc)
            return

        if self.detector.turn_is_open:
            self._utterance.append(samples)

        turn_event = self.detector.process(samples)

        if turn_event is TurnEvent.SPEECH_STARTED:
            self._utterance = [samples]
            await self.cancel_active_turn()
            await self.send(events.speech_started(self.session_id, 0))
        elif turn_event is TurnEvent.SPEECH_STOPPED:
            await self.send(events.speech_stopped(self.session_id, 0))
            utterance = np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.int16)
            self._utterance = []
            self._start_turn_task(transcript=utterance)

    def _start_turn_task(self, transcript: NDArray[np.int16] | None) -> None:
        self._turn_task = asyncio.create_task(self._run_turn(transcript))

    async def cancel_active_turn(self) -> None:
        """Barge-in: stop speaking the moment the user starts."""
        task = self._turn_task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run_turn(self, utterance: NDArray[np.int16] | None) -> None:
        """Transcribe, think and speak one turn."""
        self._response_counter += 1
        response_id = f"resp_{self._response_counter}"
        item_id = f"item_{self._response_counter}"

        try:
            if utterance is not None:
                transcript = await self.speech_to_text.transcribe(utterance, SAMPLE_RATE)
                if not transcript:
                    logger.debug("Empty transcript; turn discarded")
                    return
                await self.send(events.transcription_completed(self.session_id, transcript))
                self.messages.append({"role": "user", "content": transcript})

            await self.send(events.response_created(response_id))
            await self._stream_response(response_id, item_id)
            await self.send(events.response_done(response_id))
        except asyncio.CancelledError:
            await self.send(events.response_done(response_id, status="cancelled"))
            raise
        except httpx.HTTPError as exc:
            logger.error("0G request failed during turn: %s", exc)
            await self.send(events.error(f"0G request failed: {exc}"))
            await self.send(events.response_done(response_id, status="failed"))

    async def _stream_response(self, response_id: str, item_id: str) -> None:
        """Stream the model's answer, speaking each sentence as it completes."""
        request_messages = [{"role": "system", "content": self.instructions}, *self.messages]
        spoken_chars = 0
        full_text = ""
        tool_calls: list[ToolCall] = []

        async for stage, turn in self.language_model.stream_turn(request_messages, self.tools or None):
            full_text = turn.text
            if stage == "text":
                spoken_chars += await self._speak_complete_sentences(response_id, item_id, full_text, spoken_chars)
            elif stage == "done":
                tool_calls = turn.tool_calls

        remainder = full_text[spoken_chars:].strip()
        if remainder:
            await self._speak(response_id, item_id, remainder)

        if full_text.strip():
            self.messages.append({"role": "assistant", "content": full_text})
            await self.send(events.audio_transcript_done(response_id, item_id, full_text))

        await self.send(events.audio_done(response_id, item_id))
        await self._emit_tool_calls(response_id, item_id, tool_calls)

    async def _speak_complete_sentences(self, response_id: str, item_id: str, text: str, already_spoken: int) -> int:
        """Speak whole sentences as soon as they land, and report how much was consumed."""
        pending = text[already_spoken:]
        consumed = 0

        for sentence in _split_complete_sentences(pending):
            if len(sentence.strip()) < _MIN_SPEAKABLE_CHARS:
                continue
            await self._speak(response_id, item_id, sentence)
            consumed += len(sentence)

        return consumed

    async def _speak(self, response_id: str, item_id: str, text: str) -> None:
        """Synthesize one fragment and stream it out as audio deltas."""
        audio = await self.text_to_speech.synthesize(text)
        for start in range(0, audio.size, _AUDIO_CHUNK_SAMPLES):
            chunk = audio[start : start + _AUDIO_CHUNK_SAMPLES]
            await self.send(events.audio_delta(response_id, item_id, base64.b64encode(chunk.tobytes()).decode()))

    async def _emit_tool_calls(self, response_id: str, item_id: str, tool_calls: list[ToolCall]) -> None:
        """Forward tool calls; the client runs them and returns results as conversation items."""
        for index, call in enumerate(tool_calls):
            if not call.name:
                continue
            call_id = call.call_id or f"call_{response_id}_{index}"
            self.messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments or "{}"},
                        }
                    ],
                }
            )
            await self.send(
                events.function_call_done(response_id, item_id, call_id, call.name, call.arguments or "{}")
            )


def _split_complete_sentences(text: str) -> list[str]:
    """Return only sentences terminated by punctuation, leaving any trailing fragment behind."""
    parts = _SENTENCE_BOUNDARY.split(text)
    return parts[:-1] if len(parts) > 1 else []


def _extract_text(item: dict[str, Any]) -> str:
    """Pull plain text out of a realtime conversation item's content parts."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    return " ".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"input_text", "text"}
    ).strip()


def _to_chat_completions_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert a realtime flat tool definition into chat-completions nested form."""
    if "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        },
    }


class ZeroGRealtimeServer:
    """Serves the realtime protocol on localhost for the conversation app."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        gated: bool = False,
        language: str = "en",
        daemon_url: str = "http://localhost:8000",
    ) -> None:
        """Configure the listen address, turn trigger and robot daemon location."""
        self.host = host
        self.port = port
        self.gated = gated
        self.language = language
        self.daemon_url = daemon_url
        self.gateway = ZeroGGateway([LANGUAGE_MODEL, SPEECH_TO_TEXT])
        self._client = httpx.AsyncClient()
        self._sessions: set[RealtimeSession] = set()

    async def serve_forever(self) -> None:
        """Start the 0G proxies and accept realtime connections until cancelled."""
        self.gateway.start()

        speech_to_text = ZeroGSpeechToText(
            self.gateway.base_url(SPEECH_TO_TEXT), SPEECH_TO_TEXT.model, self._client, self.language
        )
        language_model = ZeroGLanguageModel(self.gateway.base_url(LANGUAGE_MODEL), LANGUAGE_MODEL.model, self._client)
        text_to_speech = LocalTextToSpeech(SAMPLE_RATE, piper_voice=_configured_piper_voice())

        async def handler(connection: ServerConnection) -> None:
            session = RealtimeSession(connection, speech_to_text, language_model, text_to_speech, self.gated)
            self._sessions.add(session)
            logger.info("Realtime client connected: %s", session.session_id)
            await session.send(events.session_created(session.session_id))
            try:
                async for message in connection:
                    await session.handle(message)
            except websockets.ConnectionClosed:
                logger.info("Realtime client disconnected: %s", session.session_id)
            finally:
                self._sessions.discard(session)
                await session.cancel_active_turn()

        gate_task = asyncio.create_task(self._follow_antenna_gate()) if self.gated else None
        try:
            async with websockets.serve(handler, self.host, self.port, max_size=None):
                logger.info("0G realtime server listening on ws://%s:%d/v1/realtime", self.host, self.port)
                await asyncio.Future()
        finally:
            if gate_task is not None:
                gate_task.cancel()
            await self._client.aclose()
            self.gateway.stop()

    async def _follow_antenna_gate(self) -> None:
        """Mirror antenna presses onto every live session's microphone gate."""
        gate = AntennaGate(self.daemon_url, self._client)
        transitions: asyncio.Queue[bool] = asyncio.Queue()
        poller = asyncio.create_task(gate.run(transitions))
        try:
            while True:
                is_open = await transitions.get()
                for session in self._sessions:
                    session.detector.set_gate_open(is_open)
        finally:
            poller.cancel()
