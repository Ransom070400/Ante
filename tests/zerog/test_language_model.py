"""Stream parsing and reasoning removal for the 0G language model."""

import pytest

from ante.zerog.language_model import ToolCall, strip_reasoning, _parse_stream_line, _merge_tool_call_fragment


def test_closed_reasoning_block_is_removed() -> None:
    """A provider that ignores enable_thinking must not make the robot speak its reasoning."""
    assert strip_reasoning("<think>weighing options</think>Paris is the capital.") == "Paris is the capital."


def test_reasoning_block_still_streaming_is_removed() -> None:
    """Mid-stream the closing tag has not arrived yet; nothing after <think> is speakable."""
    assert strip_reasoning("<think>I should consider") == ""


def test_text_without_reasoning_is_untouched() -> None:
    """The common case must pass through unchanged."""
    assert strip_reasoning("Paris is the capital.") == "Paris is the capital."


def test_reasoning_before_and_after_content() -> None:
    """Reasoning can precede real content in the same stream."""
    assert strip_reasoning("<think>a</think>Hello.<think>b") == "Hello."


@pytest.mark.parametrize(
    "line",
    ["", "data: [DONE]", ": keepalive", "data: not-json", 'data: {"choices": []}'],
)
def test_non_delta_lines_are_ignored(line: str) -> None:
    """Keepalives, terminators and malformed chunks must not break the turn."""
    assert _parse_stream_line(line) is None


def test_delta_line_is_parsed() -> None:
    """A normal content chunk yields its delta object."""
    assert _parse_stream_line('data: {"choices":[{"delta":{"content":"hi"}}]}') == {"content": "hi"}


def test_tool_call_arguments_are_reassembled_across_chunks() -> None:
    """Providers split tool arguments over several chunks; they must be concatenated in order."""
    calls: dict[int, ToolCall] = {}
    _merge_tool_call_fragment(calls, {"index": 0, "id": "call_1", "function": {"name": "move_head"}})
    _merge_tool_call_fragment(calls, {"index": 0, "function": {"arguments": '{"yaw":'}})
    _merge_tool_call_fragment(calls, {"index": 0, "function": {"arguments": " 30}"}})

    assert calls[0] == ToolCall(call_id="call_1", name="move_head", arguments='{"yaw": 30}')
