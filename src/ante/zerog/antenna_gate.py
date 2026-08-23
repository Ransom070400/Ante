"""Push-to-talk driven by physically moving the robot's antenna.

The antennas are motors, so their measured position doubles as a button: deflect one by
hand and the reading moves away from where the robot is holding it. That gesture opens
the microphone.

On the Wireless robot this is the cheap way to start a turn. A wake-word model would run
continuously on the CM4 for every second of silence; reading one REST value costs
effectively nothing, and it cannot be triggered by a television in the background.
"""

import math
import asyncio
import logging

import httpx


logger = logging.getLogger(__name__)

# Antennas idle within a fraction of a radian of their held position. A deliberate nudge
# moves them far past this; motor jitter and breathing animations stay well inside it.
DEFLECTION_THRESHOLD_RAD = 0.35
# The gate stays open this long after release, so it does not shut mid-sentence.
HOLD_OPEN_S = 6.0
_POLL_INTERVAL_S = 0.1
_REQUEST_TIMEOUT_S = 2.0


class AntennaGate:
    """Polls antenna position and reports whether the microphone should be open."""

    def __init__(self, daemon_url: str, client: httpx.AsyncClient) -> None:
        """Watch antennas via the daemon at ``daemon_url``."""
        self.daemon_url = daemon_url.rstrip("/")
        self._client = client
        self._resting: tuple[float, float] | None = None
        self._open_until = 0.0

    async def read_positions(self) -> tuple[float, float] | None:
        """Return (left, right) antenna angles in radians, or None when the daemon is unreachable."""
        try:
            response = await self._client.get(
                f"{self.daemon_url}/api/state/present_antenna_joint_positions", timeout=_REQUEST_TIMEOUT_S
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.debug("Antenna read failed: %s", exc)
            return None

        left, right = response.json()
        return float(left), float(right)

    def is_deflected(self, positions: tuple[float, float]) -> bool:
        """Whether either antenna has been pushed away from its resting angle."""
        if self._resting is None:
            self._resting = positions
            return False

        return any(
            abs(current - resting) > DEFLECTION_THRESHOLD_RAD for current, resting in zip(positions, self._resting)
        )

    def relax(self, positions: tuple[float, float]) -> None:
        """Re-learn the resting angle so the robot's own movements do not latch the gate open."""
        self._resting = positions

    async def run(self, on_change: "asyncio.Queue[bool] | None" = None) -> None:
        """Poll continuously, pushing gate-open transitions onto a queue."""
        loop = asyncio.get_running_loop()
        was_open = False

        while True:
            positions = await self.read_positions()
            if positions is not None:
                now = loop.time()
                if self.is_deflected(positions):
                    self._open_until = now + HOLD_OPEN_S
                elif not math.isclose(self._open_until, 0.0) and now > self._open_until:
                    self.relax(positions)

                is_open = now < self._open_until
                if is_open != was_open:
                    was_open = is_open
                    logger.info("Antenna gate %s", "opened" if is_open else "closed")
                    if on_change is not None:
                        await on_change.put(is_open)

            await asyncio.sleep(_POLL_INTERVAL_S)
