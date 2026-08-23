"""Speech-to-text through the 0G Compute Network's Whisper provider."""

import io
import wave
import logging

import httpx
import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 60.0


class ZeroGSpeechToText:
    """Transcribes PCM16 audio using a 0G-hosted Whisper model."""

    def __init__(self, base_url: str, model: str, client: httpx.AsyncClient, language: str = "en") -> None:
        """Target ``model`` through the local signing proxy at ``base_url``."""
        self.model = model
        self.base_url = base_url
        self.language = language
        self._client = client

    async def transcribe(self, samples: NDArray[np.int16], sample_rate: int) -> str:
        """Return the transcript for one utterance, or an empty string if it cannot be read."""
        audio_file = encode_wav(samples, sample_rate)

        try:
            response = await self._client.post(
                f"{self.base_url}/audio/transcriptions",
                files={"file": ("speech.wav", audio_file, "audio/wav")},
                data={"model": self.model, "language": self.language, "response_format": "json"},
                timeout=_REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("0G transcription request failed: %s", exc)
            return ""

        return str(response.json().get("text", "")).strip()


def encode_wav(samples: NDArray[np.int16], sample_rate: int) -> bytes:
    """Wrap raw PCM16 mono samples in a WAV container for multipart upload."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.astype(np.int16).tobytes())
    return buffer.getvalue()
