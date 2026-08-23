---
title: Ante
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: static
pinned: false
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Ante

A Reachy Mini you can ask anything. You talk, it answers out loud — and the model doing
the thinking runs on the **0G Compute Network**, a decentralised inference network,
rather than in any one company's data centre.

Forked from the Reachy Mini conversation app. The conversation loop, tool system,
movement manager, personality system and web UI are all inherited unchanged; what is
new here is the speech backend.

## How it works

The conversation app is an OpenAI Realtime API client. Ante therefore does not modify
that loop at all: it ships a **local server that speaks the realtime protocol** and is
backed by 0G. Point the app at `ws://127.0.0.1:8765/v1/realtime` and everything else in
the app keeps working exactly as before.

The realtime API is speech-to-speech over one socket, while 0G exposes ordinary
request/response models, so the server reassembles the three stages behind that
interface:

```
antenna / voice ─> turn detection ─> 0G  openai/whisper-large-v3   (hearing)
                                              │
                                              v
                                     0G  glm-5.2                    (thinking)
                                              │
                          ┌───────────────────┴──────────────────┐
                          v                                      v
                  Piper, on the robot                     tool calls, run by
                     (speaking)                          the app's own tools
```

**Why speech synthesis is the one local stage.** The 0G network serves chat,
transcription, image and video models, but no speech synthesis provider at all. Piper is
the deployment choice because it targets Raspberry-Pi-class ARM hardware, which is what
the Wireless robot's CM4 is.

**Why hearing is offloaded.** The Wireless robot cannot run Whisper on its CM4, so
transcription goes to the network alongside the language model.

**Why the wallet never enters this app.** Requests to 0G must be signed and settled
on-chain. Rather than reimplement that, the server supervises
`0g-compute-cli inference serve`, which owns the key and exposes a plain
OpenAI-compatible endpoint on localhost. Nothing in this codebase touches a private key
or the single-use request headers that 0G integrations usually get wrong.

## Installation

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
cp .env.example .env
```

You also need the 0G CLI, logged in with a funded account:

```bash
npm install -g @0glabs/0g-serving-broker
0g-compute-cli login
0g-compute-cli get-account
```

Both providers need a funded sub-account. Ante uses:

| Stage | Model | Provider |
|---|---|---|
| Thinking | `glm-5.2` | `0x7DCFe6AEa70350C2090041524c9B4A9262DCe87D` |
| Hearing | `openai/whisper-large-v3` | `0x36aCffCEa3CCe07cAdd1740Ad992dB16Ab324517` |

A provider with an empty sub-account returns `insufficient balance` **at request time**,
even though its proxy starts and passes its health check — a healthy proxy is not proof
of funding. Providers report this inside the body of an HTTP 200 rather than as an error
status, so Ante inspects the body and surfaces the refusal as a realtime `error` event
with the remediation command, instead of letting it read as silence.

```bash
0g-compute-cli transfer-fund --provider <address> --amount 1
```

## Running

Start the 0G backend first, then the app:

```bash
ante-0g-server            # add --push-to-talk for the antenna trigger
ante                      # in a second terminal
```

| Flag | Purpose |
|---|---|
| `--push-to-talk` | Listen only while an antenna is held, instead of always-on voice detection |
| `--daemon-url` | Robot daemon, for reading the antenna. Wireless: `http://reachy-mini.local:8000` |
| `--language` | Transcription language code (default `en`) |
| `--port` | Realtime port (default 8765; must match `HF_REALTIME_WS_URL`) |

### The antenna as a button

The antennas are motors, so their measured position doubles as a button: push one and
the reading moves away from where the robot is holding it. With `--push-to-talk`, that
gesture opens the microphone.

On the Wireless robot this is deliberately cheaper than the alternatives. A wake word
would run a model continuously on the CM4 through every second of silence, and it can be
set off by a television in the background. Reading one REST value cannot.

## Configuration

| Variable | Meaning |
|---|---|
| `HF_REALTIME_CONNECTION_MODE` | Must be `local` so the app uses this backend |
| `HF_REALTIME_WS_URL` | Where `ante-0g-server` is listening |
| `REACHY_MINI_CUSTOM_PROFILE` | Ante's personality, from `profiles/` |
| `ANTE_PIPER_VOICE` | Piper `.onnx` voice file. Without it, the server falls back to macOS `say` for laptop development |

## Known characteristics

- **A turn takes roughly seven seconds before Ante starts speaking.** This is the
  0G providers' fixed round-trip, not generation speed: measured time-to-first-token and
  total time are within 0.1 s of each other, so the response arrives in one burst.
  Ante's personality is written for short answers partly for this reason.
- **`glm-5.2` is a reasoning model.** Ante sends `enable_thinking: false`, without which
  roughly a hundred tokens and several seconds go to hidden reasoning every turn.
- **Not every provider honours that flag.** `MiniMax-M3` ignores it and streams raw
  `<think>` blocks in the content field, which the robot would read aloud, so reasoning
  is stripped before synthesis regardless of provider.

## Development

```bash
ruff check . --fix && ruff format . && mypy --pretty --show-error-codes && pytest tests/ -v
```

The 0G backend's tests live in `tests/zerog/`. The important one is
`test_live_protocol.py`: it drives the server with the real OpenAI realtime client, so if
the client library changes its event schemas, that test fails rather than the robot
failing mid-conversation.

## Customising

- Personality: `profiles/_ante_locked_profile/profile.md`
- Tools: one file per tool under `src/ante/tools/`
- Landing page: `index.html`; app settings page: `src/ante/static/index.html`

The original conversation-app README is in `README_OLD.md`.
