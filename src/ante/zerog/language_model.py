"""Streaming chat completions against a 0G Compute Network language model."""

import re
import json
import logging
from typing import Any, AsyncIterator
from dataclasses import field, dataclass

import httpx

from ante.zerog.errors import ZeroGRequestError


logger = logging.getLogger(__name__)

# glm-5.2 is a reasoning model. Left alone it spends ~100 tokens and several seconds
# thinking before the first spoken word, which is dead air in a conversation. The
# provider honours this flag by reporting reasoning_tokens=0.
_DISABLE_THINKING: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}

_REQUEST_TIMEOUT_S = 120.0

# Not every provider honours enable_thinking. MiniMax-M3 ignores it and streams raw
# <think> blocks in the content field, which the robot would otherwise read aloud, so
# reasoning is stripped from the text before it ever reaches speech synthesis.
_CLOSED_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK_BLOCK = re.compile(r"<think>.*", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove provider-leaked <think> blocks, including one still being streamed."""
    without_closed = _CLOSED_THINK_BLOCK.sub("", text)
    return _UNCLOSED_THINK_BLOCK.sub("", without_closed).strip()


@dataclass
class ToolCall:
    """A function call requested by the model, assembled from streamed fragments."""

    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamedTurn:
    """Everything the model produced for one turn."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ZeroGLanguageModel:
    """Talks to a 0G chat model through the local signing proxy."""

    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient) -> None:
        """Target ``model`` through the local signing proxy at ``base_url``."""
        self.model = model
        self.base_url = base_url
        self._client = client

    async def stream_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[tuple[str, StreamedTurn]]:
        """Yield ("text", turn) as sentences arrive and ("done", turn) at the end.

        Streaming is what makes the robot feel responsive: speech synthesis starts on the
        first complete sentence instead of waiting for the whole answer to come back.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": 500,
            **_DISABLE_THINKING,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        turn = StreamedTurn()
        raw_text = ""
        calls_by_index: dict[int, ToolCall] = {}

        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=_REQUEST_TIMEOUT_S,
        ) as response:
            response.raise_for_status()
            if "text/event-stream" not in response.headers.get("content-type", ""):
                await response.aread()
                _raise_for_error_body(response.text)

            async for line in response.aiter_lines():
                delta = _parse_stream_line(line)
                if delta is None:
                    continue

                content = delta.get("content")
                if content:
                    raw_text += content
                    spoken_text = strip_reasoning(raw_text)
                    if spoken_text != turn.text:
                        turn.text = spoken_text
                        yield "text", turn

                for fragment in delta.get("tool_calls") or []:
                    _merge_tool_call_fragment(calls_by_index, fragment)

        turn.tool_calls = [calls_by_index[index] for index in sorted(calls_by_index)]
        yield "done", turn


def _raise_for_error_body(body: str) -> None:
    """Raise when a provider answers a stream request with a refusal instead of events."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise ZeroGRequestError(f"0G provider returned an unreadable response: {body[:200]}") from None

    error = payload.get("error") if isinstance(payload, dict) else None
    if error:
        message = error.get("message") if isinstance(error, dict) else error
        raise ZeroGRequestError(str(message))


def _parse_stream_line(line: str) -> dict[str, Any] | None:
    """Return the delta object from one SSE line, or None for keepalives and terminators."""
    line = line.strip()
    if not line.startswith("data:"):
        return None

    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None

    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        logger.warning("Discarding malformed stream chunk: %s", data[:200])
        return None

    choices = chunk.get("choices") or []
    if not choices:
        return None

    delta = choices[0].get("delta")
    return delta if isinstance(delta, dict) else None


def _merge_tool_call_fragment(calls_by_index: dict[int, ToolCall], fragment: dict[str, Any]) -> None:
    """Accumulate a streamed tool-call fragment; arguments arrive split across chunks."""
    index = fragment.get("index", 0)
    call = calls_by_index.setdefault(index, ToolCall())

    if fragment.get("id"):
        call.call_id = fragment["id"]

    function = fragment.get("function") or {}
    if function.get("name"):
        call.name = function["name"]
    if function.get("arguments"):
        call.arguments += function["arguments"]
