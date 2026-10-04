# PersonaPlex Duplex Preload

[NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) 7B speech-to-speech model via a custom duplex server built on `moshi.server`.

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | Deploy | Hugging Face token (accept [model license](https://huggingface.co/nvidia/personaplex-7b-v1) first) |

## Deploy (Modal)

Create a Modal secret named `hf-token` with key `HF_TOKEN`, then:

```bash
uv run modal deploy models/personaplex/deploy/personaplex.py
```

The deployment pins NVIDIA PersonaPlex commit `3428dfd95309a7f3c84fd93259ded0f810d1ff91` and runs the custom duplex server instead of stock `moshi.server`.

Runtime endpoint is fixed in `model.toml`:

`https://peoplemakethings--people-bench-personaplex-serve.modal.run`

## Interact

```bash
uv run bench interact --model personaplex
```

Preload a processed eval point, then continue live:

```bash
uv run bench interact --model personaplex --preload-point data/processed/PMT_0005/points/000001
```

`--preload-point` also loads `point.json` `text_prompt` as the session system
prompt unless `--text-prompt` is passed.

Override session defaults from `model.toml`:

```bash
uv run bench interact --model personaplex --text-prompt "You enjoy having a good conversation."
uv run bench interact --model personaplex --voice-prompt NATF2.pt
```

## Duplex wire protocol

Stock chat UI and live conversation still use `/api/chat`:

- `\x00` handshake
- `\x01` + Opus live user audio in
- `\x02` + UTF-8 streamed text out

Duplex history priming uses `/api/duplex` on the same host:

- `\x03` session metadata JSON (`voice_prompt`, `text_prompt`, optional `seed`)
- `\x04` user history PCM16 mono @ 24 kHz
- `\x05` assistant history PCM16 mono @ 24 kHz
- `\x06` assistant transcript JSON with word timestamps
- `\x07` commit prefill
- `\x08` primed acknowledgement JSON
- then live `\x01` / `\x02` traffic continues on the same socket

History audio must be an exact multiple of 1,920 samples (80 ms @ 24 kHz). The server teacher-forces synchronized user audio, assistant audio, and frame-aligned assistant text through `LMGen.step(...)` without resetting streaming state.

## Prepare eval points

```bash
uv run prepare-eval --output data/processed
uv run prepare-eval example_1 --output data/processed
```

Reviewed labels are read from `s3://pmt-data-annotated/reviewed/`, channel transcripts
from `s3://pmt-data-annotated/drafts/`, and source audio from `s3://pmt-data-raw/`
using the `pmt` AWS profile.

Each eval point writes `assistant_transcript.json` (Whisper word timestamps over
interact `assistant.wav`) and `rollout_assistant_transcript.json` (same shape
over eval history `rollout_assistant.wav`). These are timed words
(`text` / `start_sec` / `end_sec`), not freeform paragraphs.

Eval duplex path for PersonaPlex: preload `rollout_user_context.wav` +
`rollout_assistant.wav` (+ transcript) into both history channels, then stream
the live user channel. Pause points stream `rollout_user_input_pause_end.wav`
followed by frame-aligned silence for the drain; EOT points stream
`rollout_user_input.wav` followed by frame-aligned silence until the model stops
speaking. Both are streamed rather than waited out because the model steps once
per received frame and would otherwise never run inside the window being
measured. `rollout.wav` in results is only that live mix — see root `README.md`
(“How eval feeds each model family”) and `AGENTS.md`.

`context.json` holds conversation context events (completed turns as text and
the current user turn as base64 PCM16 audio when draft transcripts are available).

## Offline prefill proof (Modal GPU)

```bash
uv run modal run models/personaplex/deploy/experiment.py
```

This runs a GPU-side teacher-forcing smoke test and writes `prefill_proof.json` to the `people-bench-artifacts` Modal volume.

## Forced-text rank probe (Modal GPU)

```bash
uv run modal run models/personaplex/deploy/rank_probe.py::probe
```

Voice prompt x history-length factorial over the rank the text head gives the
token prefill forces on it, plus a silence continuation per arm. Uploads each
arm's history through `bench.history.condition_duplex_history`, writes
`rank_probe.json` to the volume and to `data/analysis/voice_prompt/`. See
`docs/phonation.md`.

## Prefill dose experiment (Modal GPU)

```bash
uv run modal run models/personaplex/deploy/dose.py::dose --seeds 5
```

Three prefill arms per distinct duplex history — unmasked, voiced-but-unworded
assistant frames masked to silence, and the same amount of voiced audio masked
from frames a word does cover — with several seeds per cell, since prefill is
deterministic and the continuation is where the warm-up is measured. Writes
`dose_probe.json`; analyse with `data/analysis/dose/dose_analysis.py`.

## Known limitations

- Cold start on Modal can take several minutes after scale-to-zero
- One active duplex session per GPU container (`max_inputs=1`); eval overlaps
  up to 64 containers (`max_containers=64`, `max_concurrency=64`). Idle
  containers scale to zero after 2 minutes (`min_containers=0`)
- Assistant transcript alignment is best-effort from word timestamps; empty transcripts still preload audio-only history
- History is capped at `PERSONAPLEX_MAX_HISTORY_FRAMES` (2048 frames, 163.84 s)
  and truncated from the front. That is NVIDIA's trained window
  ([paper](https://arxiv.org/abs/2602.06053),
  [explainability.md](https://huggingface.co/nvidia/personaplex-7b-v1/blob/main/explainability.md)),
  not Moshi's 3000-frame inference ring. The drop count is reported in the
  primed payload and lands in `trace.json` as a `duplex_primed` event
- Eval sends no voice prompt, so the session's voice comes from the
  teacher-forced assistant history; `interact` still accepts `--voice-prompt`

## Seeding

The bundled duplex server seeds torch, numpy, and random per session when the
metadata carries a `seed`. `bench eval` sends `BASE_SEED + rollout_index` so
each rollout is reproducible while rollouts still differ; override the base with
`[session] seed` in `model.toml`, or set it to `-1` to send no seed at all.
