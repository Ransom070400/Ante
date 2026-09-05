"""Tests for app-level runtime behavior."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from pytest import MonkeyPatch

import ante.main as main_mod


def test_inactivity_timeout_thread_goes_to_sleep() -> None:
    """The watchdog should use the shared sleep shutdown path once activity is too old."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())
    go_to_sleep = MagicMock(return_value={"status": "sleeping"})

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
        go_to_sleep=go_to_sleep,
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    go_to_sleep.assert_called_once_with()
    stream_manager.close.assert_not_called()


def test_inactivity_timeout_thread_closes_stream_manager_without_sleep_callback() -> None:
    """The watchdog should still close the stream when no sleep callback is available."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    stream_manager.close.assert_called_once_with()


def test_controller_zero_g_backend_reuses_open_port(monkeypatch: MonkeyPatch) -> None:
    """The app-controller launch path should not spawn a duplicate backend."""
    logger = MagicMock()
    monkeypatch.setattr(main_mod, "_is_port_open", lambda host, port: True)
    popen = MagicMock()
    monkeypatch.setattr(main_mod.subprocess, "Popen", popen)

    process = main_mod._start_controller_zero_g_backend(
        "127.0.0.1",
        8765,
        "en",
        "http://localhost:8000",
        logger,
    )

    assert process is None
    popen.assert_not_called()
    logger.info.assert_called_once_with("Reusing 0G realtime backend on %s:%d", "127.0.0.1", 8765)


def test_controller_zero_g_backend_starts_local_backend(monkeypatch: MonkeyPatch) -> None:
    """The app-controller launch path should start the backend module when absent."""
    logger = MagicMock()
    process = MagicMock()
    monkeypatch.setattr(main_mod, "_is_port_open", lambda host, port: False)
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(main_mod.subprocess, "Popen", popen)

    result = main_mod._start_controller_zero_g_backend(
        "127.0.0.1",
        8765,
        "fr",
        "http://localhost:8000",
        logger,
    )

    assert result is process
    popen.assert_called_once_with(
        [
            main_mod.sys.executable,
            "-m",
            "ante.zerog",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--language",
            "fr",
            "--daemon-url",
            "http://localhost:8000",
        ]
    )
