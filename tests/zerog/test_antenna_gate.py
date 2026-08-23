"""Antenna deflection must open the gate without latching on the robot's own motion."""

import httpx
import pytest

from ante.zerog.antenna_gate import DEFLECTION_THRESHOLD_RAD, AntennaGate


@pytest.fixture
def gate() -> AntennaGate:
    """Return a gate pointed at a daemon that is never actually contacted."""
    return AntennaGate("http://localhost:8000", httpx.AsyncClient())


def test_first_reading_only_learns_the_resting_pose(gate: AntennaGate) -> None:
    """Nothing is a deflection until there is a baseline to deflect from."""
    assert gate.is_deflected((0.0, 0.0)) is False


def test_small_movement_is_not_a_press(gate: AntennaGate) -> None:
    """Motor jitter and idle animations must not open the microphone."""
    gate.is_deflected((0.0, 0.0))
    assert gate.is_deflected((DEFLECTION_THRESHOLD_RAD / 2, 0.0)) is False


def test_deliberate_deflection_opens_the_gate(gate: AntennaGate) -> None:
    """A hand pushing the antenna crosses the threshold."""
    gate.is_deflected((0.0, 0.0))
    assert gate.is_deflected((DEFLECTION_THRESHOLD_RAD * 2, 0.0)) is True


def test_either_antenna_counts(gate: AntennaGate) -> None:
    """Both antennas are valid buttons."""
    gate.is_deflected((0.0, 0.0))
    assert gate.is_deflected((0.0, -DEFLECTION_THRESHOLD_RAD * 2)) is True


def test_relaxing_adopts_a_new_resting_pose(gate: AntennaGate) -> None:
    """After the robot repositions its antennas, that becomes the new baseline."""
    gate.is_deflected((0.0, 0.0))
    moved = (DEFLECTION_THRESHOLD_RAD * 2, 0.0)
    assert gate.is_deflected(moved) is True
    gate.relax(moved)
    assert gate.is_deflected(moved) is False


@pytest.mark.asyncio
async def test_unreachable_daemon_returns_none() -> None:
    """A missing robot degrades to no gate rather than crashing the server."""
    gate = AntennaGate("http://127.0.0.1:9", httpx.AsyncClient())
    assert await gate.read_positions() is None
