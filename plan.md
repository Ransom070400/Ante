# Ante — Reachy Mini voice assistant on 0G Compute

## 1. What I understand you want

A Reachy Mini app you can talk to conversationally — ask it anything out loud,
and it answers out loud, with the robot moving expressively while it speaks.
The "brain" (the LLM) must run on the **0G Compute Network** decentralised
inference API rather than a centralised provider like OpenAI.

## 2. Findings from environment inspection

Already installed and working on this machine:

| Thing | Status |
|---|---|
| `reachy-mini-app-assistant` | installed (`~/.local/bin`) |
| `uv` 0.12.5, Python 3.14.6, Node 22.20, `git-lfs` 3.7.1 | present |
| `0g-compute-cli` 0.7.5 | present, logged in |
| 0G wallet | `~/.0g-compute-cli/config.json`, **mainnet** |

**Live 0G inference providers** (queried, not assumed). 22 providers. Relevant ones:

| Model | Provider | Verifiability | Your sub-account balance |
|---|---|---|---|
| `glm-5.2` | `0x7DCFe6AEa70350C2090041524c9B4A9262DCe87D` | TeeML | **1.531 0G, no withdrawal pending** |
| `MiniMax-M3` | `0xa6581CfDc65278cC539e94d864012ce4B35c5D56` | TeeML | **1.800 0G, no withdrawal pending** |
| `openai/whisper-large-v3` | `0x36aCffCEa3CCe07cAdd1740Ad992dB16Ab324517` | TeeML | 0.000 0G — needs funding to use |
| `qwen3.6-plus` | `0x992e6396157Dc4f22E74F2231235D7DE62696db5` | TeeML | 1.5 but full withdrawal requested |
| `claude-fable-5` | `0x1F444c8A8D0b8e99A50e9f165806d28B01916E04` | standard | 2.308 but full withdrawal requested |

Two consequences that shape the design:

1. **0G can do speech-to-text too** — `whisper-large-v3` is on the network. So
   the listening half can also be decentralised, not just the thinking half.
2. **0G has no text-to-speech provider.** The voice output *must* be local.
   This is a hard constraint, not a preference.

## 3. App flavour: Python (deliberate deviation from the JS default)

AGENTS.md says default to a JS app. I'm proposing **Python** here, because this
app hits three of the four stated exceptions:

- **Secrets**: 0G requests are signed with a wallet private key. That key can
  never go into browser JS — it would be handed to every visitor. Signing has
  to happen in a trusted process.
- **Heavy on-robot compute**: local TTS (and optionally local STT) is a real
  audio pipeline, not a UI.
- **Real-time loop**: mic capture → VAD → STT → LLM → TTS → motion needs a
  deterministic loop with barge-in handling.

Scaffold with the **`conversation` template**, which already ships the mic/VAD
plumbing, tool dispatch, movement manager and Gradio UI:

```
reachy-mini-app-assistant create --template conversation ante <path> --publish
```

Known gotcha: the scaffolder refuses to run inside an existing git repo, so it
scaffolds to a temp dir and the contents get moved into place.

## 4. Technical approach

The conversation template is built around **one** integration seam:
`src/ante/openai_realtime.py`, defining `OpenaiRealtimeHandler`
(a `fastrtc.AsyncStreamHandler`). OpenAI's realtime API is a single socket doing
STT+LLM+TTS at once; 0G is a plain OpenAI-compatible *chat completions* endpoint.
So that one class gets replaced by a three-stage pipeline, and everything else in
the template (tool dispatch, Gradio UI, movement manager, profiles) is untouched.

```
mic ──> VAD ──> [STT] ──> [LLM: 0G chat completions] ──> [TTS] ──> speaker
                                      │
                                      └── tool calls ──> move queue ──> control loop
```

Per-turn 0G call sequence (the settle step is what actually pays the provider):

```
broker.inference.getServiceMetadata(provider)      -> { endpoint, model }
broker.inference.getRequestHeaders(provider, q)    -> single-use auth headers
POST {endpoint}/chat/completions                   -> the answer
broker.inference.processResponse(provider, chatId) -> settle payment
```

Two things I will get right rather than discover the hard way:

- **Headers are single-use.** They must be regenerated per request; reusing them
  across turns fails. This is the most common 0G integration bug.
- **The signing broker is TypeScript** (`@0glabs/0g-serving-broker`), while the
  Reachy app is Python. Planned resolution: a small local Node sidecar process
  that owns the key and exposes an OpenAI-shaped endpoint on `127.0.0.1`, which
  the Python app calls. This also keeps the key out of the app's process space.
  (Alternative, if you'd rather have no Node dependency: reimplement the header
  signing in Python with `eth-account`. More code, one less moving part.)

Robot behaviour while talking: antenna motion driven by speech amplitude, a
"thinking" pose while the 0G round-trip is in flight, and emotion playback from
`pollen-robotics/reachy-mini-emotions-library` chosen by an LLM tool call.

## 5. Security note (raising once, then proceeding)

`~/.0g-compute-cli/config.json` holds a **mainnet** private key with real funds
and is currently mode `644` — world-readable by any process or user on this
machine. I'd like to `chmod 600` it. The app will read the key from env/config
at runtime and never copy it into the repo; `.env` goes in `.gitignore`.

## 6. Decisions (answered)

**Q1. "cover with"** → Resolved by building the union of both readings. The app is
a spoken conversational assistant (ask anything, it answers aloud), and the
trigger to start a turn is **covering / touching the antenna**. Both readings of
the original request are satisfied by one design.

**Q2. Robot** → **Wireless** (onboard CM4, antennas). Confirmed by user.

**Q3. Brain** → **`glm-5.2`**, provider `0x7DCFe6AEa70350C2090041524c9B4A9262DCe87D`.
Chosen because it has a funded sub-account with no withdrawal pending (1.531 0G),
is TeeML-verifiable, and has reliable tool-calling — which the motion/emotion
tools depend on.

**Q4. STT** → **0G `whisper-large-v3`**, provider `0x36aCffCEa3CCe07cAdd1740Ad992dB16Ab324517`.
Not a preference — a consequence of Q2. The CM4 cannot run local Whisper, so
offloading STT is required. Sub-account is at 0 balance and **needs funding
before first use**; I will confirm with you before moving real mainnet funds.

**Q5. Turn trigger** → **Antenna touch/cover** (antennas double as buttons).
Deliberately chosen over always-on VAD or a wake word, both of which would run a
model continuously on a compute-constrained CM4. Antenna touch costs zero CPU.

## 7. Consequences of Wireless for the audio pipeline

Everything that can be offloaded, is. Only TTS stays local, because 0G has no
TTS provider at all (verified against the live provider list — the network has
STT, chat, image and video models, but no speech synthesis).

```
antenna cover ──> mic capture ──> 0G whisper-large-v3   (offloaded)
                                          │
                                          v
                               0G glm-5.2 chat          (offloaded)
                                          │
                        ┌─────────────────┴────────────┐
                        v                              v
                 Piper TTS (local, CM4)          tool calls ──> move queue
                        │                                            │
                        v                                            v
                 robot speaker  <──── antenna motion driven by amplitude
```

**Piper** is the TTS choice: it is built for Raspberry-Pi-class ARM hardware,
which is exactly what the CM4 is. Heavier neural TTS would not hold real-time on
this robot.

---

# Outcome

Built and verified on 2026-08-23. Full quality gate green:
`ruff check .` clean, `mypy --strict` clean over 56 files, **372 tests passing**,
and `reachy-mini-app-assistant check .` passes all checks.

## What was built

The conversation app is an OpenAI Realtime API client. Rather than rewrite its
conversation loop, Ante ships a **local server that speaks the realtime protocol**
and is backed by 0G. The app is pointed at it by two lines of `.env`. Its tool
system, movement manager, personality system and web UI are untouched.

New code, all under `src/ante/zerog/`:

| Module | Role |
|---|---|
| `realtime_server.py` | The realtime protocol server and turn pipeline |
| `events.py` | Server-event builders, field-exact to the client's schemas |
| `gateway.py` | Supervises `0g-compute-cli inference serve` proxies |
| `language_model.py` | Streaming 0G chat + tool calls + reasoning stripping |
| `speech_to_text.py` | 0G Whisper transcription |
| `text_to_speech.py` | Local Piper, with a macOS backend for laptop development |
| `voice_activity.py` | Turn detection, optionally gated |
| `antenna_gate.py` | Antenna-as-button push-to-talk |

## Verified against the live network

- A real `AsyncOpenAI` realtime client completes a full turn against the server and
  parses **every** event emitted (`tests/zerog/test_live_protocol.py`).
- A real end-to-end turn against live 0G mainnet produced, in Ante's own voice:
  *"A model called GLM, served over the 0G Compute Network. No single company owns
  the machine that's thinking right now."* — synthesized to 7.58 s of speech.
- Tool calling works on `glm-5.2` and returns well-formed arguments.

## Findings that changed the design

1. **~7 s fixed latency per turn.** Time-to-first-token and total time are within
   0.1 s of each other on both funded providers, so responses arrive in one burst;
   this is network round-trip, not generation speed. Ante's personality is written
   for short answers partly because of this.
2. **`glm-5.2` is a reasoning model.** Without `enable_thinking: false` it spends
   ~108 of 144 tokens and several seconds on hidden reasoning per turn.
3. **`MiniMax-M3` ignores that flag** and streams raw `<think>` blocks in the content
   field, which the robot would speak aloud. Reasoning is now stripped defensively
   regardless of provider. This is why `glm-5.2` is the chosen brain, on evidence.
4. **0G has no speech synthesis provider,** so TTS is necessarily local.

## Scaffolder defects found and worked around

The installed `reachy-mini` 1.9.0 scaffolder has drifted from the current template:

1. It clones a `develop` branch that no longer exists upstream, so
   `create --template conversation` fails outright. Worked around by driving its own
   code path with the branch patched to `main`, leaving the installed package unmodified.
2. It writes the locked profile to `src/<app>/profiles/`, but the template now resolves
   profiles from the project root. Moved to `profiles/`.
3. It emits the old profile format (`instructions.txt` + `tools.txt`); the template now
   uses a single `profile.md` with a TOML front-matter tool list. Rewritten.
4. Its `LOCKED_PROFILE` setting breaks 30 inherited tests, which assume profile
   switching stays available. Ante selects its personality with
   `REACHY_MINI_CUSTOM_PROFILE` instead — same default personality, green gate, and the
   personality picker keeps working.

These are worth reporting upstream; item 1 makes the documented command unusable.

## Closed: full speech-to-speech verified on live 0G mainnet

Verified 2026-08-24 once the Whisper sub-account was funded. Two complete turns,
microphone audio in and synthesized speech out, every stage on the live network:

| Heard (0G whisper-large-v3) | Replied (0G glm-5.2) |
|---|---|
| "Hello robot, what is the capital of France?" | "Paris." |
| "what are you actually running on, and keep it brief." | "A model called GLM, served over the 0G Compute Network. No single company owns the machine that's thinking right now." |

**Measured turn latency: ~13 s** from end of speech to first audio — about 5 s for
transcription and 8 s for the reply. The two calls are inherently serial. This is the
providers' fixed round-trip, not generation speed.

**Funding gotcha found in the process**: a provider requires
`minimum reserve + unsettled fees`, and the reserve is a locked floor that cannot be
spent from. Funding a sub-account with exactly 1.0 0G bought exactly one request, then
refused everything after with `required minimum is 1.003071 0G`. Fund with headroom
above the reserve. The proxy auto-funds the difference once the ledger can cover it, but
only picks up a new balance on restart.

## Superseded: funding speech-to-text

Transcription is the one thing not yet exercised end to end. Provider
`0x36aCff…` requires **1.0 0G locked** in its sub-account and currently holds zero;
its proxy starts and passes a health check, then returns `insufficient balance` at
request time.

The ledger holds 7.29 0G but only **0.154 0G is available** — the rest is locked in
sub-accounts. Two of those already have full withdrawals requested (`qwen3.6-plus`
1.5 0G, `claude-fable-5` 2.308 0G), so roughly 3.8 0G can be reclaimed rather than
depositing new funds. This is a real on-chain transaction on mainnet, so it is left
for the user to authorise.
