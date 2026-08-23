"""The Piper voice must come from the environment, and a bad path must not be fatal."""

from pathlib import Path

import pytest

from ante.zerog.realtime_server import PIPER_VOICE_ENV, _configured_piper_voice


def test_unset_voice_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no voice configured the server falls back to its development backend."""
    monkeypatch.delenv(PIPER_VOICE_ENV, raising=False)
    assert _configured_piper_voice() is None


def test_missing_voice_file_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in the path must degrade to the fallback rather than crash the robot."""
    monkeypatch.setenv(PIPER_VOICE_ENV, "/nonexistent/voice.onnx")
    assert _configured_piper_voice() is None


def test_existing_voice_file_is_used(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A real voice file is passed through to synthesis."""
    voice = tmp_path / "en_US-lessac-medium.onnx"
    voice.write_bytes(b"")
    monkeypatch.setenv(PIPER_VOICE_ENV, str(voice))
    assert _configured_piper_voice() == voice
