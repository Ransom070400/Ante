"""Turn detection for the 0G realtime backend."""

import logging
from enum import Enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class TurnEvent(Enum):
    """What the detector concluded about the audio it just saw."""

    NONE = "none"
    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"


@dataclass
class VoiceActivityConfig:
    """Thresholds for energy-based turn detection."""

    # RMS of int16 samples. Speech in a quiet room sits well above this; fan noise sits below.
    speech_threshold: float = 500.0
    # Silence needed to end a turn. Long enough to survive the pause between clauses.
    silence_duration_s: float = 0.8
    # Speech needed to start a turn, so a cough or a door does not open one.
    min_speech_duration_s: float = 0.20


class VoiceActivityDetector:
    """Detects turn boundaries in a PCM16 stream, optionally gated by a push-to-talk signal.

    Gating matters on the Wireless robot: an always-on wake-word model would burn CM4
    cycles continuously, whereas the antenna gate costs nothing until it is touched.
    """

    def __init__(self, sample_rate: int, config: VoiceActivityConfig | None = None, gated: bool = False) -> None:
        """Detect turns in a ``sample_rate`` stream, gated when ``gated`` is set."""
        self.gated = gated
        self.sample_rate = sample_rate
        self.config = config or VoiceActivityConfig()

        self._is_open = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._gate_is_open = not gated

    def set_gate_open(self, is_open: bool) -> None:
        """Open or close the push-to-talk gate; closing it ends any turn in progress."""
        if not self.gated:
            return
        self._gate_is_open = is_open
        if not is_open:
            self._speech_samples = 0

    @property
    def turn_is_open(self) -> bool:
        """Whether a user turn is currently in progress."""
        return self._is_open

    def process(self, samples: NDArray[np.int16]) -> TurnEvent:
        """Feed one audio frame and return any turn boundary it crossed."""
        if not self._gate_is_open and not self._is_open:
            return TurnEvent.NONE

        is_speech = _rms(samples) > self.config.speech_threshold

        if is_speech:
            self._silence_samples = 0
            self._speech_samples += samples.size
            if not self._is_open and self._speech_samples >= self._samples_for(self.config.min_speech_duration_s):
                self._is_open = True
                return TurnEvent.SPEECH_STARTED
            return TurnEvent.NONE

        if not self._is_open:
            self._speech_samples = 0
            return TurnEvent.NONE

        self._silence_samples += samples.size
        if self._silence_samples >= self._samples_for(self.config.silence_duration_s):
            self._is_open = False
            self._speech_samples = 0
            self._silence_samples = 0
            return TurnEvent.SPEECH_STOPPED

        return TurnEvent.NONE

    def reset(self) -> None:
        """Drop any in-progress turn, used when a session restarts."""
        self._is_open = False
        self._speech_samples = 0
        self._silence_samples = 0

    def _samples_for(self, duration_s: float) -> int:
        return int(duration_s * self.sample_rate)


def _rms(samples: NDArray[np.int16]) -> float:
    """Root-mean-square level of an int16 frame."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
