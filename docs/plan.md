# people-bench: Multi-Model Eval Plan

Reference plan for turning people-bench from a PersonaPlex-only deploy into an extensible
realtime voice eval that others can contribute models and scenarios to.

## Current State

The repo today is a minimal PersonaPlex-on-Modal deployment with a live WebSocket terminal
client. There is no eval infrastructure, model abstraction, or automated scoring yet.

```
people-bench/
├── env.py                     # .env loader
├── modal_app/personaplex.py   # Modal GPU deployment
├── bench/test_live.py         # Live mic/speaker client (PersonaPlex-specific)
└── pyproject.toml
```

Key gaps:

- PersonaPlex wire protocol (`\x00/\x01/\x02`, Opus framing) is hardcoded in the client
- No model registry or adapter pattern
- No tasksets, metrics, headless mode, or CI-friendly tests
- `bench/test_live.py` is a manual smoke test, not an automated benchmark

---

## Target Architecture

Three layers: **model registry**, **interact**, and **eval**.

```
                    ┌─────────────────────────────────┐
                    │  bench interact --model <id>    │  human testing
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  bench eval <taskset> --model   │  automated eval
                    │         (verifiers harness)      │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  RealtimeModel protocol          │
                    │  (connect, send/recv audio/text) │
                    └──────────────┬──────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
   personaplex/            openai-realtime/           gemini-live/
   model.toml + adapter     model.toml + adapter       model.toml + adapter
```

### Layer 1: Model Registry

Every model — hosted (OpenAI Realtime, Gemini) or self-hosted (PersonaPlex, Moshi, Qwen
Omni) — gets the same folder structure:

```
models/
  personaplex/
    model.toml      # connection, audio format, session params, secrets
    adapter.py      # implements RealtimeModel
    README.md       # setup instructions for contributors
    deploy/         # optional: Modal/Docker deploy scripts
  openai-realtime/
    model.toml
    adapter.py
  gemini-live/
    ...
  qwen-omni/
    ...
```

#### Model config (`model.toml`)

Each model config defines connection, audio format, session parameters, and secret env var
names. Hosted and self-hosted models differ mainly in `connection` and `session`.

**Self-hosted example (PersonaPlex):**

```toml
[model]
id = "personaplex"
name = "NVIDIA PersonaPlex 7B"
modality = "speech-to-speech"

[connection]
type = "websocket"
url = "https://peoplemakethings--people-bench-personaplex-serve.modal.run"

[audio]
input_sample_rate = 24000
output_sample_rate = 24000
encoding = "opus"
frame_samples = 1920

[session]
voice_prompt = "NATF2.pt"
text_prompt = ""

[secrets]
hf_token_env = "HF_TOKEN"
```

**Hosted example (OpenAI Realtime):**

```toml
[model]
id = "openai-realtime"
name = "OpenAI Realtime API"
modality = "speech-to-speech"

[connection]
type = "websocket"
url = "wss://api.openai.com/v1/realtime"
api_key_env = "OPENAI_API_KEY"

[audio]
input_sample_rate = 24000
output_sample_rate = 24000
encoding = "pcm16"

[session]
model = "gpt-4o-realtime-preview"
voice = "alloy"
instructions = "You are a helpful assistant."
```

#### RealtimeModel protocol

The only interface shared by interact and eval. Protocol-specific details stay inside each
adapter.

```python
class RealtimeModel(Protocol):
    def endpoint(self, session: SessionConfig) -> str: ...
    async def connect(self, session: SessionConfig) -> None: ...
    async def send_audio(self, pcm: np.ndarray) -> None: ...
    async def recv_audio(self) -> AsyncIterator[np.ndarray]: ...
    async def recv_text(self) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...

    @property
    def metrics(self) -> SessionMetrics: ...  # TTFB, turn latency, etc.
```

### Layer 2: Interact (human testing)

Thin CLI that loads a model config, instantiates the adapter, and runs a mic/speaker loop.

```bash
uv run bench interact --model personaplex
uv run bench interact --model openai-realtime
uv run bench interact --model @ models/personaplex/model.toml
```

Refactor `bench/test_live.py` into `bench/interact.py`. The interact CLI should only know
about `RealtimeModel`, not PersonaPlex.

### Layer 3: Eval (automated)

```
bench/
  interact.py
  tasksets/
    empathy_v1/            # scenarios + reward functions
    turn_taking_v1/
  harness/
    voice_loop.py          # synthetic user, recorded audio, or LLM user-sim
  metrics/
    latency.py
    transcript.py
configs/
  evals/
    empathy-v1.toml        # taskset + model + rollout settings
```

```bash
uv run bench eval empathy-v1 --model personaplex -n 50
uv run bench eval @ configs/evals/turn-taking-v1.toml --model openai-realtime
```

Tasks are structured scenarios: e.g. play an audio clip, expect a response within a latency
budget, score with a rubric. Headless mode feeds WAV/Opus instead of a live mic for CI.

---

## Migration from Current Code

| Today | Becomes |
|-------|---------|
| `bench/test_live.py` wire protocol + Opus | `models/personaplex/adapter.py` |
| `PERSONAPLEX_URL` in `.env` | fixed URL in `models/personaplex/model.toml` + generic `--model` selector |
| `modal_app/personaplex.py` | `models/personaplex/deploy/modal.py` |
| `env.py` | stays; resolves `*_env` keys from model configs |

---

## Prime Intellect Verifiers Integration

**Use verifiers for the eval layer, not for model wiring.**

### Good fit

- **Tasksets** — scenarios, rubrics, `@reward` / `@metric`
- **Traces** — full rollout records (timing, transcripts, scores)
- **Distribution** — Environments Hub (`prime eval run your-org/people-empathy-v1`)
- **Concurrency** — bounded parallel rollouts (`-n 50 -r 3`)
- **Training path** — same env config can feed RL later via prime-rl

### Does not fit out of the box

- Realtime bidirectional audio (default harnesses are chat/API)
- Per-model wire protocols (PersonaPlex vs OpenAI Realtime vs Gemini Live)
- Latency metrics like time-to-first-audio-byte

### Recommended split

| Layer | Owner |
|-------|--------|
| Model configs + adapters | people-bench (model registry) |
| Voice rollout loop | Custom verifiers harness (`voice_loop`) |
| Scenarios + scoring | Verifiers taskset |
| CLI / Hub / traces | Verifiers eval |

Verifiers v1 is designed for custom harnesses when the rollout loop is not standard chat.
The voice harness would:

1. Load a `RealtimeModel` from the registry
2. Run the turn loop (user audio in, model audio/text out)
3. Emit a `Trace` with transcripts and timing for taskset rewards

Hosted models plug in via adapters; verifiers does not need to know their APIs.

---

## Contributor Contract

### Adding a model

1. `models/<id>/model.toml` — required fields (documented in a schema)
2. `models/<id>/adapter.py` — implements `RealtimeModel`
3. `models/<id>/README.md` — env vars, deploy steps, known limitations
4. Optional: `models/<id>/deploy/` — Modal/Docker for self-hosted models
5. Smoke test: `uv run bench interact --model <id> --headless --duration 5s`

### Adding an eval scenario

1. `bench/tasksets/<name>/` — tasks + rewards
2. `configs/evals/<name>.toml` — wires taskset + default models

---

## Implementation Phases

### Phase 1: Model registry + interact

- Define `RealtimeModel` protocol and `SessionMetrics`
- Extract PersonaPlex into `models/personaplex/` (config + adapter)
- Add `bench/interact --model <id>`
- Move `modal_app/personaplex.py` to `models/personaplex/deploy/modal.py`
- Prove the adapter pattern with PersonaPlex before adding a second model

**Exit criteria:** `uv run bench interact --model personaplex` works identically to today's
`test_live.py`.

### Phase 2: Headless eval runner

- One taskset with recorded-audio scenarios
- Basic metrics: latency (TTFB, turn duration), transcript capture
- Headless mode (no mic/speaker) for CI
- `bench/eval` CLI without verifiers dependency yet

**Exit criteria:** `uv run bench eval smoke-v1 --model personaplex -n 5` produces scored
traces locally.

### Phase 3: Verifiers integration + Hub

- Wrap taskset as a verifiers v1 environment
- Custom `voice_loop` harness
- Publish to Environments Hub
- Add a second model adapter (e.g. OpenAI Realtime) to validate the registry

**Exit criteria:** `prime eval run your-org/people-smoke-v1 --model personaplex` works;
contributors can add models via the registry contract.

---

## Design Principles

1. **Unify the interface, not the wire protocol.** Each model adapter owns its transport
   details; interact and eval only speak `RealtimeModel`.
2. **Config over code for model parameters.** Every model fills out `model.toml`; adapters
   implement connection and framing only.
3. **Interact first, eval second.** Human smoke testing should never depend on verifiers.
4. **Headless by default for CI.** Live mic/speaker is for local dev; eval runs on recorded
   audio.
5. **Minimal scope per PR.** Phase 1 should not block on verifiers or a second model.

---

## Open Questions

- [ ] Schema validation for `model.toml` (pydantic model vs JSON Schema)
- [ ] Whether model registry auto-discovers `models/*/model.toml` or uses explicit registration
- [ ] Scoring approach for subjective voice quality (LLM judge vs human rubric vs ASR + text eval)
- [ ] Warmup / cold-start handling for self-hosted Modal models in timed evals
- [ ] Whether to keep `bench/test_live.py` as a deprecated alias during Phase 1 migration
