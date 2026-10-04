# people-bench

Benchmark and interact with realtime speech-to-speech models from the terminal. The repo has a shared model registry, a live mic/speaker client, and a data pipeline that builds eval points from annotated conversations in S3.

Supported models today: [PersonaPlex](https://github.com/NVIDIA/personaplex) and [Kyutai Moshi](https://github.com/kyutai-labs/moshi) (self-hosted on Modal), OpenAI GPT Realtime, Gemini 3.1 Live, and xAI Grok Voice.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [Modal](https://modal.com/) account for PersonaPlex / Moshi deploy (`modal setup`)
- Hugging Face account with the [PersonaPlex license](https://huggingface.co/nvidia/personaplex-7b-v1) accepted (Moshi weights are public)
- OpenAI API key for GPT Realtime and `prepare-eval` speaker personas
- Gemini API key (`GCP_API_KEY`) for Gemini Live
- xAI API key (`XAI_API_KEY`) for Grok Voice
- AWS CLI profile `pmt` for `prepare-eval` (reads annotated data from S3)

## Setup

```bash
cp .env.example .env
# Set HF_TOKEN, OPENAI_API_KEY, GCP_API_KEY, and XAI_API_KEY in the main checkout.

uv sync
```

Git worktrees load that same main-checkout `.env`; they do not need their own copy.

All local packages (`bench/`, `data_processing/`, `scripts/`) are installed editably, so code changes apply without reinstalling. After changing dependencies in `pyproject.toml`, run:

```bash
uv sync --reinstall-package people-bench
```

| Variable | Required for | Purpose |
|----------|--------------|---------|
| `HF_TOKEN` | PersonaPlex / Moshi deploy | Hugging Face token for model weights |
| `OPENAI_API_KEY` | GPT Realtime, `prepare-eval` | Realtime API access and speaker persona generation |
| `GCP_API_KEY` | Gemini Live | Gemini API key for the Live API |
| `XAI_API_KEY` | Grok Voice | xAI API key for the Voice Agent API |
| `PERSONA_MODEL` | `prepare-eval` | Optional OpenAI model for speaker personas (default `gpt-4o`) |

## Live interaction

```bash
uv run bench interact --model personaplex
uv run bench interact --model moshi
uv run bench interact --model gpt-realtime-fast
uv run bench interact --model gpt-realtime-slow
uv run bench interact --model gemini-live
uv run bench interact --model grok-voice
```

Useful flags:

```bash
uv run bench interact --model personaplex --list-devices
uv run bench interact --model personaplex --input-device 0 --output-device 1
uv run bench interact --model personaplex --voice-prompt NATF2.pt --text-prompt "You enjoy conversation."
uv run bench interact --model gpt-realtime-fast --voice-prompt alloy
uv run bench interact --model gemini-live --voice-prompt Puck
uv run bench interact --model grok-voice --voice-prompt eve
```

Press Ctrl+C to stop.

### Preload conversation context

Processed eval points live under `data/processed/<source_id>/points/<NNNNNN>/`. Each point includes role-separated audio, metadata, and optional `context.json`.

Models declare which context types they support. The interact runner picks the first matching type from the model's priority list and errors if none match.

| Context type | Contents | PersonaPlex / Moshi | GPT Realtime | Gemini Live | Grok Voice |
|--------------|----------|---------------------|--------------|-------------|------------|
| `text` | Completed turns as text events | no | yes | yes | yes |
| `text_audio` | Text turns + trailing user PCM16 audio | no | yes (preferred) | yes (preferred) | yes (preferred) |
| `duplex_audio` | Synchronized user/assistant PCM + transcript | yes | no | no | no |

```bash
uv run bench interact --model personaplex \
  --preload-point data/processed/example_1/points/000004

uv run bench interact --model gpt-realtime-fast \
  --preload-point data/processed/example_1/points/000003

uv run bench interact --model gemini-live \
  --preload-point data/processed/example_1/points/000003

uv run bench interact --model grok-voice \
  --preload-point data/processed/example_1/points/000003
```

See `models/personaplex/README.md`, `models/moshi/README.md`, `models/gpt-realtime/README.md`, `models/gemini-live/README.md`, and `models/grok-voice/README.md` for protocol and config details.

## Prepare eval points

Build eval points from reviewed annotations and raw channel audio in S3:

```bash
uv run prepare-eval --output data/processed
uv run prepare-eval example_1 --output data/processed
uv run prepare-eval --datasource seamless --limit 2 --output data/processed
```

`--datasource` selects the ingest adapter (default `annotated`). Each adapter
normalizes to the same `label.json` + `waves/speaker_{1,2}.wav` layout before
points are written, so eval artifacts stay identical across sources.

| `--datasource` | Conversations | Audio / labels |
|----------------|---------------|----------------|
| `annotated` (default) | `s3://pmt-data-annotated/reviewed/<datapoint>/label.json` plus drafts and `s3://pmt-data-raw/<datapoint>/channel_{1,2}.wav` | Already in bench speaker IDs |
| `seamless` | `s3://pmt-data-seamless/datasets/v1/sessions/<session>/ground_truth.json` plus the two speaker WAVs it points at | Converted A/B → `speaker_1`/`speaker_2`, 48 kHz float → 24 kHz PCM16 |

`--limit N` processes the first N conversations. `--max-points N` writes at most
N pause/EOT points per conversation (and skips recall), useful for a smoke review.
`--recall-only` skips pause/EOT and writes only the recall points. If the
datapoint already exists, those points are kept and recall is appended.

Besides each conversation's own `manifest.json`, prepare-eval maintains one
cross-source manifest at the output root (`data/processed/manifest.json`)
listing every point across conversations, so a multi-source eval is a single
`bench eval --manifest data/processed/manifest.json`. A fresh output gets a
new manifest; when the output already holds one from earlier runs, only the
just-processed sources' entries are replaced, so the rest survive re-runs of
a subset.

### Curating split events

Long sources label far more split events than an eval needs (the seamless set
has ~46k across 176 sessions). Curation scores every event and keeps the few
that make fair, hard eval points:

```bash
uv run prepare-eval --datasource seamless --curate-only            # heuristic pass
uv run modal run data_processing/curation_modal.py                 # same, S3-local (fast)
uv run prepare-eval --judge-only                                   # LLM judge re-rank
```

The heuristic pass applies hard gates (pause length, turn context, reference
adjacency and substance, and audio checks through the same `bench/speech.py`
gate scoring uses) and ranks survivors, keeping `--keep-per-type` per session
(default 3). `--judge-only` then runs a gpt-4o judge over the survivors'
transcript windows: it rejects incoherent windows (scripted stimulus
readings), pauses that hand the turn to the listener, interrupted fragments,
and unresponsive references, and re-ranks the rest by temptation/difficulty.
An event the judge scores zero is never selected, however few candidates a
session has.

Decisions live in `data/curation/<source_id>.json` (features, judge verdict,
keep/reject with a stable reason) with figures next to them (`curation.png`,
`curation_judge.png`). A normal `prepare-eval` run applies the decision file
automatically when one exists for a source and stamps the kept point's
`point.json` with the curation score and features.

Output layout per point:

```
data/processed/example_1/
  personas.json               # per-speaker identity prompts (see below)
  manifest.json
  points/000004/
    point.json                # split metadata, file references, assistant text_prompt
    user.wav                  # active speaker, 0 → pause_end (or EOT); interact duplex
    assistant.wav             # other speaker, same length and frame alignment
    assistant_transcript.json # Whisper timestamps for assistant.wav (interact)
    context.json              # conversation context events (see below)
    rollout_user_context.wav  # user history before the active turn (duplex eval)
    rollout_assistant.wav     # assistant history, same interval (duplex eval)
    rollout_assistant_transcript.json  # Whisper timestamps for rollout_assistant
    rollout_context.json      # completed turns as text (GPT / Gemini / Grok / non-duplex eval)
    rollout_user_input_pause_start.wav  # pause: turn start → pause_start (non-duplex eval live)
    rollout_user_input_pause_end.wav    # pause: turn start → pause_end (duplex eval live)
    rollout_user_input.wav              # EOT only: turn start → turn end
    rollout_reference_response.wav      # EOT only: human assistant's next turn
    rollout_reference_response_transcript.json
```

`prepare-eval` writes both `pause_start` and `end_of_turn` points (same
ordering as annotations), then conversation-recall and fact-recall points for
each 30 / 60 / 90 / 120 s prefix that has a completed turn. Recall needs
`uv sync --extra tts` (Qwen3-TTS; keeps `sentence-transformers` on 5.x
because that extra pins `transformers==4.57`). A missing TTS extra fails the
run: an environment that cannot synthesize probes must not silently produce a
dataset with no recall points (use `--max-points` for a recall-free smoke
run). Extracted conversation facts pass an LLM judge for specificity before a
probe is synthesized, and probes that leak a literal speaker label are
dropped. EOT points without a following
assistant turn are skipped. Re-run `prepare-eval` to pick up
`rollout_reference_response.*` on existing datasets.

When draft transcripts are available, `prepare-eval` also asks OpenAI for two
short **identity prompts** (one per speaker): who they are, how they know the
other person, and speaking style. It does **not** ask for answers or a recap.
Each prompt is wrapped with a fixed frame telling the model that assistant
history is its own prior speech and not to mention being an AI. The matching
prompt is stored on `point.json` as `text_prompt` for whoever is
`assistant_speaker` on that point. Eval (and `interact --preload-point`) pass
that into the session. Without transcripts, points still get the identity
frame alone.

Audio invariants: 24 kHz sample rate, mono, sample count a multiple of 1920 (80 ms PersonaPlex frame).

### How eval feeds each model family

Eval always has two phases: **history preload**, then **live stream**. Both the
history representation and the live protocol differ by family: a duplex model
runs one LM step per received frame, so it has to be streamed the pause (and
the EOT drain) rather than left to wait either out.

**Full duplex (PersonaPlex, Moshi)** — two parallel audio history channels, then live mic:

1. **History preload** (before scoring starts):
   - User channel: `rollout_user_context.wav` (everything before the active turn)
   - Assistant channel: `rollout_assistant.wav` (same interval, frame-aligned)
   - Alignment: `rollout_assistant_transcript.json` — Whisper **word timestamps**
     (`{text, start_sec, end_sec}`), not a freeform paragraph. Teacher-forced with
     the assistant audio during prefill.
2. **Live stream, pause points:** `rollout_user_input_pause_end.wav` — the
   active user turn plus the real recorded pause, then frame-aligned silence for
   the drain. Streaming the pause is what lets the model be stepped, and so
   speak, inside the window being scored.
3. **Live stream, EOT points:** `rollout_user_input.wav`, then frame-aligned
   silence until the model stops talking. Same reason: a wall-clock wait would
   leave the model unstepped for the response being measured.

`user.wav` / `assistant.wav` / `assistant_transcript.json` are the **interact**
preload of the full `0 → boundary` window. They are **not** the eval history path.

**Non-duplex / text context (GPT Realtime, Gemini Live, Grok Voice):**

1. **History preload:** `rollout_context.json` — completed turns as text events
   (no duplex WAVs).
2. **Live stream, pause points:** `rollout_user_input_pause_start.wav`, then a
   wall-clock wait through the labeled pause. Server-side VAD runs on its own
   clock, so waiting is correct here.
3. **Live stream, EOT points:** the full user turn, then the buffer is committed
   and a response requested, then a wall-clock wait until the model is done.
   Server VAD is off for EOT so mid-turn pauses cannot interrupt.

`rollout.wav` in results is a **listening mix of the live phase only** (left =
streamed user input, right = model output). History is never mixed into it —
short `rollout.wav` for PersonaPlex or Moshi does not mean history was skipped. See
`AGENTS.md` for the full artifact contract.

### context.json schema

Top-level object with an `events` array. Each event is a `conversation.item.create` with this shape:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "message",
    "role": "user",
    "content": [{"type": "input_text", "text": "hello"}]
  }
}
```

Content types:

| type | role | meaning |
|------|------|---------|
| `input_text` | `user` | completed user turn (text) |
| `output_text` | `assistant` | completed assistant turn (text) |
| `input_audio` | `user` | current user turn audio (base64 PCM16 at 24 kHz) with optional `transcript` |

Classification:

- **text** — events contain only text content types
- **text_audio** — events include one trailing `input_audio` item for the active user turn

`prepare-eval` writes `text_audio` contexts when draft channel transcripts are available in S3.

Splits whose timestamp exceeds source audio duration are skipped with a log message.

## Run deterministic rollouts

Replay prepared active turns at their original timing and score pause
recognition, EOT response length, and recall. EOT rollouts also write a
similarity sidecar (`token F1` + embedding cosine) next to the length score.
Omit `--eval-type` to run every metric present in the manifest, or pass
`pause-recognition` / `response-length` / `conversation-recall` /
`fact-recall` to run one:

```bash
uv run bench eval \
  --model gpt-realtime-fast gpt-realtime-slow gemini-live grok-voice personaplex moshi \
  --manifest data/processed/example_1/manifest.json \
  --rollouts 3

# Several manifests in one batch (keeps the GPU pool busy across chunks):
uv run bench eval \
  --model all \
  --manifest data/processed/seamless_V03_S2060_I00000197_*/manifest.json \
  --rollouts 1

uv run bench eval \
  --model gpt-realtime-fast \
  --manifest data/processed/example_1/manifest.json \
  --eval-type pause-recognition \
  --rollouts 3

uv run bench eval \
  --model gpt-realtime-fast \
  --manifest data/processed/example_1/manifest.json \
  --eval-type response-length \
  --rollouts 3

uv run bench eval \
  --model gpt-realtime-fast \
  --point data/processed/example_1/points/000001 \
  --rollouts 1
```

History + live wiring differs by model family (see **How eval feeds each model
family** above). Either way the pause scoring window is
`pause_start → pause_end`, and any model speech overlapping it is a false
takeover.

For EOT points the live stream is `rollout_user_input.wav`. The harness then
drains until the model finishes talking — streaming silence frames for duplex
models, waiting on the wall clock for hosted ones — and compares that reply to
the human assistant's next turn (duration, Whisper word count, and
token F1 / embedding cosine on that same transcript pair).

Recall points use that same live drain. History is a 30 / 60 / 90 / 120 s
prefix; the live turn is a synthesized probe in `speaker_2`'s voice.
`conversation-recall` asks about a fact from the first 30 s;
`fact-recall` asks a fixed world-knowledge question. Both are binary
(transcript vs gold). Probe audio is generated at prepare time (`uv sync --extra tts`).

Each rollout writes `trace.json`, `score.json`, `response.wav`, and a stereo
`rollout.wav` listening mix of the **live phase only** (left = streamed user
input, right = model output). History is not in that file. Duration can differ
across models when one keeps speaking after the live input ends.

`bench eval` also mirrors every artifact to
`s3://pmt-data-seamless/results/<run_id>/` with the same layout as local disk:
each rollout directory uploads as soon as it finishes, and the run-level
summaries and figures upload when the run completes.

A run lands in `data/results/<timestamp>/` (override with `--output`):
top-level `results.json` plus `pause.png` / `length.png` / `length-average.png` / `recall.png` /
`similarity.png` / `latency.png` across evals, then one folder per live eval
(`pause-recognition`, `response-length`, `conversation-recall`,
`fact-recall`) with its own `results.json` and a subfolder per model
(that is where the per-model figure lives). EOT rollouts live under
`response-length/` and write `similarity.json` next to `score.json`.
`score.json` carries
the model id so `uv run bench report data/results/<run_id>` can rebuild
figures without re-running eval. `rollout_count` is scored rollouts only;
`unscored` and `errors` sit beside the means so a broken rollout cannot pass
as a patient or a terse one. See `AGENTS.md` for artifact semantics.

Eval sends **no voice prompt**: the teacher-forced assistant history already
establishes the voice, and a stock prompt would contradict both it and the
persona. `bench interact` still takes `--voice-prompt`.

Rollout N is seeded `BASE_SEED + N` so it replays identically while still
differing from its siblings. Override the base with `[session] seed` in
`model.toml`, or `-1` to send no seed.

Triage a finished run without listening to anything:

```bash
uv run bench analyze data/results/<run_id>
```

This flags rollouts where the model consumed fewer frames than were streamed,
where input streaming fell behind schedule, where the response is a spectrally
frozen drone, or where `score.json` disagrees with its own trace. See
**Verifying the audio pipeline without listening** in `AGENTS.md`.

## Deploy PersonaPlex / Moshi

Create a Modal secret named `hf-token` with key `HF_TOKEN`, then:

```bash
uv run modal deploy models/personaplex/deploy/personaplex.py
uv run modal deploy models/moshi/deploy/serve.py
```

Stable URLs:

- PersonaPlex: `https://peoplemakethings--people-bench-personaplex-serve.modal.run`
- Moshi: `https://peoplemakethings--people-bench-moshi-serve.modal.run`

Both use the custom duplex server in `models/personaplex/server/` (`/api/duplex` history priming, `/api/chat` live). Moshi loads official Kyutai weights (`kyutai/moshiko-pytorch-bf16`) on that same server. GPU scales to zero after 2 minutes idle — cold starts can take several minutes. Eval warms the duplex pool before the first rollout so that wait sits outside the timed run.

Ephemeral smoke deploy:

```bash
uv run modal run models/personaplex/deploy/personaplex.py
```

Offline GPU prefill proof:

```bash
uv run modal run models/personaplex/deploy/experiment.py
```

## Project layout

```
bench/                          Shared runtime
  interact.py                   Live mic/speaker session
  protocol.py                   RealtimeModel interface and context types
  registry.py                   Model discovery and adapter loading
  points.py                     Load processed eval points for preload
  transcribe.py                 Whisper transcription for prepare-eval and scoring
  speech.py                     Shared RMS speech detection and EOT onset
  audio_metrics.py              Objective PCM measurements (see AGENTS.md)
  trace.py                      TraceEvent, RolloutTrace, score-file status
  metrics/
    pause.py                    Pause-recognition scorer
    length.py                   EOT response-length scorer
    recall.py                   Conversation / fact recall scorer
    similarity.py               EOT token F1 + embedding cosine
  report/
    artifacts.py                Load score.json / error.json from a run
    summarize.py                Rebuild results.json
    plots.py                    pause.png / length.png / length-average.png / recall.png / similarity.png / latency.png
    analyze.py                  Flag suspect rollouts from their artifacts

data_processing/                Eval point preparation (S3 → local)
  prepare_eval.py               Build role-separated points from labels + audio
  personas.py                   Identity system prompts from channel transcripts
  labels.py                     Parse turn-taking split events
  context.py                    Build context.json event payloads
  recall.py                     Conversation / fact recall point writer
  tts.py                        Qwen3-TTS voice clone for recall probes

models/                         One folder per model
  personaplex/
    model.toml                  Connection, audio, session defaults
    adapter.py                  WebSocket client (chat + duplex preload)
    server/                     Duplex PersonaPlex server (bundled to Modal)
      wire.py                   Duplex preload wire protocol helpers
    deploy/                     Modal deploy and offline prefill proof
  moshi/
    model.toml                  Kyutai Moshi (shared PersonaPlex adapter)
    deploy/                     Modal deploy of official Kyutai weights
  gpt-realtime/
    adapter.py                  Shared OpenAI Realtime client
  gpt-realtime-fast/
    model.toml                  300 ms VAD profile
  gpt-realtime-slow/
    model.toml                  500 ms VAD profile
  gemini-live/
    model.toml                  Gemini 3.1 Flash Live
    adapter.py                  Live API WebSocket client
  grok-voice/
    model.toml                  xAI Grok Voice (shared GPT Realtime adapter)

scripts/interact.py             `bench` CLI entrypoint
scripts/interact_with_history.py  Example text-context preload for GPT Realtime
tests/                          Unit tests for context builders and adapters
docs/plan.md                    Architecture roadmap (automated eval, verifiers)
```

## Adding a model

1. Add `models/<id>/model.toml` with connection, audio, and session settings.
2. Implement `models/<id>/adapter.py` with `create(config) -> RealtimeModel`.
3. Smoke test: `uv run bench interact --model <id>`.

Use `models/personaplex/` as the reference for self-hosted WebSocket models, `models/moshi/` for a second duplex model on that same path, and `models/gpt-realtime/`, `models/gemini-live/`, or `models/grok-voice/` for hosted APIs.

## Tests

```bash
uv sync --extra dev
uv run pytest tests/
```

This includes a GPU-free loopback of the whole PersonaPlex path (real
websockets, real Opus codec, real server coroutines against a fake model) and
objective audio checks in `bench/audio_metrics.py`. `AGENTS.md` explains what
each suite proves and what a failure means.
