"""0G Compute Network provider registry for the Ante voice assistant."""

from typing import Final
from dataclasses import dataclass


@dataclass(frozen=True)
class ZeroGProvider:
    """A single 0G Compute Network inference provider."""

    address: str
    model: str
    port: int


# Verified against `0g-compute-cli inference list-providers` on 0G mainnet (chain 16661).
# The language model is a reasoning model, so ANTE disables thinking per request in
# language_model.py; leaving it on costs ~100 wasted tokens and seconds of latency per turn.
LANGUAGE_MODEL: Final = ZeroGProvider(
    address="0x7DCFe6AEa70350C2090041524c9B4A9262DCe87D",
    model="glm-5.2",
    port=3001,
)

# 0G is the only speech-to-text option here: the Wireless robot's CM4 cannot run
# Whisper locally, so transcription is offloaded to the network like the language model.
SPEECH_TO_TEXT: Final = ZeroGProvider(
    address="0x36aCffCEa3CCe07cAdd1740Ad992dB16Ab324517",
    model="openai/whisper-large-v3",
    port=3002,
)

# The network has no speech synthesis provider at all (it serves chat, transcription,
# image and video models), so text-to-speech is the one stage that must stay local.
