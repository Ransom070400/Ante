"""Builders for the OpenAI Realtime server events that the conversation loop consumes.

Field sets here are not guesswork: they mirror the required fields of the
``openai.types.realtime`` models, and ``tests/zerog/test_events.py`` validates every
builder against the SDK's own parser so a client-side schema bump fails loudly here.
"""

import uuid
from typing import Any


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex[:24]}"


def session_created(session_id: str) -> dict[str, Any]:
    """Handshake event the client waits for before sending session.update."""
    return {
        "event_id": _event_id(),
        "type": "session.created",
        "session": {"id": session_id, "type": "realtime", "object": "realtime.session"},
    }


def session_updated(session_id: str) -> dict[str, Any]:
    """Acknowledge an applied session.update."""
    return {
        "event_id": _event_id(),
        "type": "session.updated",
        "session": {"id": session_id, "type": "realtime", "object": "realtime.session"},
    }


def speech_started(item_id: str, audio_start_ms: int) -> dict[str, Any]:
    """Signal that the user began speaking; drives the robot's listening pose."""
    return {
        "event_id": _event_id(),
        "type": "input_audio_buffer.speech_started",
        "audio_start_ms": audio_start_ms,
        "item_id": item_id,
    }


def speech_stopped(item_id: str, audio_end_ms: int) -> dict[str, Any]:
    """Signal end of the user's turn."""
    return {
        "event_id": _event_id(),
        "type": "input_audio_buffer.speech_stopped",
        "audio_end_ms": audio_end_ms,
        "item_id": item_id,
    }


def transcription_completed(item_id: str, transcript: str) -> dict[str, Any]:
    """Deliver the final user transcript. The usage block is required by the client schema."""
    return {
        "event_id": _event_id(),
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
        "usage": {"type": "tokens", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def response_created(response_id: str) -> dict[str, Any]:
    """Open a response; the client treats this as "assistant is speaking"."""
    return {
        "event_id": _event_id(),
        "type": "response.created",
        "response": {"id": response_id, "object": "realtime.response", "status": "in_progress", "output": []},
    }


def response_done(response_id: str, status: str = "completed") -> dict[str, Any]:
    """Close a response and release the client's single-response lock."""
    return {
        "event_id": _event_id(),
        "type": "response.done",
        "response": {"id": response_id, "object": "realtime.response", "status": status, "output": []},
    }


def audio_delta(response_id: str, item_id: str, base64_pcm: str) -> dict[str, Any]:
    """Stream one chunk of synthesized speech as base64 PCM16."""
    return {
        "event_id": _event_id(),
        "type": "response.output_audio.delta",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": base64_pcm,
    }


def audio_done(response_id: str, item_id: str) -> dict[str, Any]:
    """Mark the end of synthesized speech for a response."""
    return {
        "event_id": _event_id(),
        "type": "response.output_audio.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
    }


def audio_transcript_done(response_id: str, item_id: str, transcript: str) -> dict[str, Any]:
    """Publish what the assistant said, for the transcript panel."""
    return {
        "event_id": _event_id(),
        "type": "response.output_audio_transcript.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "content_index": 0,
        "transcript": transcript,
    }


def function_call_done(response_id: str, item_id: str, call_id: str, name: str, arguments: str) -> dict[str, Any]:
    """Hand a tool call to the client, which runs it through its background tool manager."""
    return {
        "event_id": _event_id(),
        "type": "response.function_call_arguments.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": 0,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def error(message: str, code: str = "server_error") -> dict[str, Any]:
    """Surface a failure to the client instead of dropping the turn silently."""
    return {
        "event_id": _event_id(),
        "type": "error",
        "error": {"type": "invalid_request_error", "code": code, "message": message},
    }
