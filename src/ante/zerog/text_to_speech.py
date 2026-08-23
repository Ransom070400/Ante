"""Local speech synthesis.

This is the one stage that cannot be offloaded: the 0G network serves chat,
transcription, image and video models, but no speech synthesis provider. Piper is the
deployment target because it is built for Raspberry-Pi-class ARM hardware, which is
exactly the Wireless robot's CM4. The macOS backend exists so the app can be developed
and tested on a laptop without cross-compiling Piper.
"""

import wave
import shutil
import asyncio
import logging
import tempfile
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

PIPER_BINARY = "piper"
MACOS_BINARY = "say"


class TextToSpeechError(RuntimeError):
    """Raised when no synthesis backend is available."""


class LocalTextToSpeech:
    """Synthesizes speech to PCM16 at the conversation's sample rate."""

    def __init__(self, sample_rate: int, piper_voice: Path | None = None, macos_voice: str | None = None) -> None:
        """Select a synthesis backend and emit audio at ``sample_rate``."""
        self.sample_rate = sample_rate
        self.piper_voice = piper_voice
        self.macos_voice = macos_voice

        if piper_voice is not None and shutil.which(PIPER_BINARY):
            self._backend = "piper"
        elif shutil.which(MACOS_BINARY):
            self._backend = "macos"
        else:
            raise TextToSpeechError(
                "No speech synthesis backend. Install piper and set ANTE_PIPER_VOICE to a .onnx voice file."
            )
        logger.info("Speech synthesis backend: %s", self._backend)

    async def synthesize(self, text: str) -> NDArray[np.int16]:
        """Return spoken audio for a sentence, or an empty array when synthesis fails."""
        if not text.strip():
            return np.zeros(0, dtype=np.int16)

        try:
            if self._backend == "piper":
                return await self._synthesize_piper(text)
            return await self._synthesize_macos(text)
        except (OSError, subprocess.SubprocessError, wave.Error) as exc:
            logger.error("Speech synthesis failed for %r: %s", text[:60], exc)
            return np.zeros(0, dtype=np.int16)

    async def _synthesize_piper(self, text: str) -> NDArray[np.int16]:
        """Piper writes raw PCM16 at its voice's native rate on stdout."""
        process = await asyncio.create_subprocess_exec(
            PIPER_BINARY,
            "--model",
            str(self.piper_voice),
            "--output_raw",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        raw, stderr = await process.communicate(text.encode())
        if process.returncode != 0:
            logger.error("piper exited %s: %s", process.returncode, stderr.decode(errors="replace")[:200])
            return np.zeros(0, dtype=np.int16)

        return resample(np.frombuffer(raw, dtype=np.int16), _piper_voice_rate(self.piper_voice), self.sample_rate)

    async def _synthesize_macos(self, text: str) -> NDArray[np.int16]:
        """`say` cannot write to stdout, so it goes through a temporary WAV file."""
        with tempfile.TemporaryDirectory() as workdir:
            output = Path(workdir) / "speech.wav"
            command = [MACOS_BINARY, "-o", str(output), "--data-format=LEI16@16000"]
            if self.macos_voice:
                command += ["-v", self.macos_voice]
            command.append(text)

            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error("say exited %s: %s", process.returncode, stderr.decode(errors="replace")[:200])
                return np.zeros(0, dtype=np.int16)

            with wave.open(str(output), "rb") as wav_file:
                samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
                return resample(samples, wav_file.getframerate(), self.sample_rate)


def _piper_voice_rate(voice_path: Path | None) -> int:
    """Piper voices are named like en_US-lessac-medium.onnx; 22050 Hz is the common rate."""
    return 22050


def resample(samples: NDArray[np.int16], source_rate: int, target_rate: int) -> NDArray[np.int16]:
    """Linear-interpolation resample, adequate for speech and free of a SciPy dependency."""
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.int16)

    target_length = int(round(samples.size * target_rate / source_rate))
    source_positions = np.linspace(0, samples.size - 1, num=target_length)
    resampled = np.interp(source_positions, np.arange(samples.size), samples.astype(np.float32))
    return resampled.astype(np.int16)
