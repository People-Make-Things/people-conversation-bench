# Agent / human reference: eval point artifacts

This document defines what each audio/context file means, which models consume
it, and what the rollout outputs represent. Read this before comparing results
across models.

## Point directory layout

Prepared under `data/processed/<source_id>/`:

| File | What it is |
|------|------------|
| `personas.json` | Per-speaker identity prompts generated at prepare time, plus generator model / prompt version / transcript hash. |

Prepared under `data/processed/<source_id>/points/<NNNNNN>/`:

| File | What it is |
|------|------------|
| `user.wav` | Active speaker audio from conversation start through the focused boundary (`pause_end` for pause points, turn end for EOT). Same length as `assistant.wav`. |
| `assistant.wav` | Other speaker audio over that same interval. |
| `assistant_transcript.json` | Whisper word timestamps for `assistant.wav` (interact duplex preload). |
| `context.json` | Full interact preload as GPT-style conversation events (`text` / `text_audio`). |
| `rollout_user_context.wav` | User audio **before** the active turn (history / context preload), capped at the shared history budget. |
| `rollout_assistant.wav` | Assistant audio over that same history interval. Equal length and frame-aligned with `rollout_user_context.wav`. |
| `rollout_assistant_transcript.json` | Whisper word timestamps for `rollout_assistant.wav` (eval duplex preload). Loaded as-is. |
| `rollout_context.json` | Completed turns inside the same capped history window, as text events (no live turn audio). |
| `rollout_user_input_pause_start.wav` | Pause points only: live user turn from turn start through annotated `pause_start`. Kept for checkpoint validation; no longer streamed by eval. |
| `rollout_user_input_pause_end.wav` | Pause points only: same turn through annotated `pause_end` (speech + real pause audio). **Streamed by both eval families**: duplex needs real frames through the pause, and hosted server VAD needs to hear the silence. |
| `rollout_user_input.wav` | EOT: live user turn through turn end. Recall: synthesized probe in the user's voice. |
| `rollout_reference_response.wav` | EOT points only: human assistant's next turn after the split (eval reference). |
| `rollout_reference_response_transcript.json` | Whisper word timestamps for `rollout_reference_response.wav`. |
| `point.json` | Metadata, speaker roles, rollout timing (`checkpoint_sec`, `window_end_sec`, …), and `text_prompt` for the current `assistant_speaker`. |

Audio invariants: 24 kHz mono PCM16; sample counts for duplex history are a
multiple of 1920 (80 ms PersonaPlex frame).

Pause / EOT history is capped at `HISTORY_CAP_SEC` (163.84 s, PersonaPlex's
2048-frame trained window) at prepare time, cut at a completed-turn boundary
so `rollout_user_context.wav` / `rollout_assistant.wav` and
`rollout_context.json` describe the same turns. The point on disk is therefore
exactly what every model conditions on — the server-side truncation in
`models/personaplex/server/prefill.py` never fires on prepared points, and
hosted models see the same capped window as text. The cut is recorded as
`rollout.history_start_sec` in `point.json`. Recall points build their own
30–120 s prefixes and are untouched.

## Curation (which split events become points)

`data/curation/<source_id>.json` records a keep/reject decision per labeled
split event: hard gates (pause length, turn context, EOT reference adjacency
`gap_sec <= 3` and substance, audio validated through the same `bench/speech.py`
gate scoring uses), a heuristic score, and a gpt-4o judge verdict over the
transcript window (`data_processing/curation_judge.py`). The judge rejects
scripted stimulus readings, pauses whose last words hand the turn to the
listener, interrupted user-turn fragments, and non-responsive references; a
judge score of zero is never selected. `prepare-eval` applies the decision
file automatically when present and stamps `point.json` with `curation`.
Re-ranking is resumable: judged verdicts are stored per event and keyed by
judge version + model.

## Who uses what at eval time

Models declare `supported_contexts`. The harness loads matching artifacts and
forks:

### Duplex models (PersonaPlex, Moshi)

1. **History preload:** `rollout_user_context.wav` + `rollout_assistant.wav` +
 `rollout_assistant_transcript.json` as two parallel mono history streams.
2. **Live stream, pause points:** `rollout_user_input_pause_end.wav` in full,
 then `pause_drain_sec` of frame-aligned silence.
3. **Live stream, EOT and recall points:** `rollout_user_input.wav` in full, then
 frame-aligned silence until the model stops speaking (or the hard cap).

Everything is streamed on `stream_pcm`'s absolute-deadline schedule, so wall
clock still advances at realtime and `observed_input_time` maps onto frames
that were really sent.

A duplex model runs one LM step per received frame: it has no clock of its own.
The harness used to stream only through `pause_start` and then sleep out the
pause, which meant the model was never stepped inside `[checkpoint,
window_end]` and could not have spoken there however much it wanted to. Every
rollout drifted toward a pass for a reason unrelated to turn-taking. **Pause
scores produced before this change are not comparable with scores after it.**

The EOT drain has the same shape: a wall-clock wait after the user turn would
leave a duplex model unstepped for the whole response it is being measured on,
so the drain is streamed as real silence frames rather than slept through.
Because those frames are traced as `input_audio_sent`, the `out/in` ratio
`bench analyze` reports stays meaningful across the drain.

`user.wav` / `assistant.wav` are for interact preload of the full
`0 → split` window, not the eval rollout history path.

### Non-duplex / text-context models (GPT Realtime, Gemini Live, Grok Voice)

1. **History preload:** `rollout_context.json` — completed turns as text
   (transcripts were produced at prepare time; eval does not re-transcribe
   `rollout_user_context.wav`).
2. **Live stream, pause points:** `rollout_user_input_pause_end.wav` in full,
   then `pause_drain_sec` of real silence frames — the same stream the duplex
   path gets. Server VAD only closes the user turn on silence it receives, so
   the earlier stop-at-pause_start-and-wait protocol meant no response was
   ever created and 98–99% of hosted pause rollouts came back `no_model_audio`.
   **Hosted pause scores produced before this change are not comparable with
   scores after it.**
3. **Live stream, EOT and recall points:** the full user turn (or synthesized
 probe), then `end_input()` (commit the buffer and request a response), then
 wait on the wall clock until the model stops speaking or the hard cap trips.
 Server VAD is off so mid-turn pauses cannot interrupt the model.

GPT Realtime, Gemini Live, and Grok Voice never consume the duplex WAV history
pair during eval. Moshi does — same path as PersonaPlex. The pause hold is one
shared path (both families stream the pause and real silence drain frames); the
live protocol forks by model family only for the EOT drain: server-side VAD
runs on its own clock, so waiting is the right thing for a hosted model and the
wrong thing for a step-driven duplex model. The fork keys off the context the
model selected (`ContextType.DUPLEX_AUDIO`), not a separate flag. The stop
*condition* is shared — both families stop on the same speech definition from
`bench/speech.py`; only the way they wait differs.

Eval and interact preload send `point.json` `text_prompt` as the model session
instructions (GPT Realtime / Grok Voice `instructions`, Gemini Live
`systemInstruction`, PersonaPlex `<system>` prompt). This is the identity frame
plus the bio for `assistant_speaker`, so injected assistant history is treated as
the model's own prior speech.

## Timing for pause points

Relative to the live turn start (`rollout_user_input_*` start):

- `checkpoint_sec` — annotated `pause_start` (≈ duration of
  `rollout_user_input_pause_start.wav`)
- `window_end_sec` — annotated `pause_end` (≈ duration of
  `rollout_user_input_pause_end.wav`)
- Both families: harness streams the pause_end WAV, which carries the real
  pause audio across `checkpoint` → `window_end`, then `pause_drain_sec` of
  silence frames (duplex needs the LM steps; hosted VAD needs to hear the
  silence)
- A model that has produced no audio when the fixed drain ends gets a bounded
  extension (`PAUSE_NO_AUDIO_EXTENSION_SEC`) of further silence frames, plus a
  short onset tail once audio appears. Grok regularly delivers first audio
  ~0.6 s after `response_created`, which the fixed drain was cutting off and
  recording as `no_model_audio`.

Score = pass if the model does **not** utter words whose audio overlaps the
pause hold (`checkpoint` → `window_end`). Above-gate murmur with an empty text
stream is not a takeover. Speech that starts only after `window_end_sec` is
not a false takeover.

A rollout that returns **no** model audio is not a pass. `score.json` records
`status: no_model_audio` with `score: null`, and `results.json` counts it under
`unscored` rather than folding it into `mean_score`. Silence only counts as good
turn-taking when the model was demonstrably alive and generating it.

## Timing for EOT points

Relative to the live turn start (`rollout_user_input.wav` start):

- `checkpoint_sec` / `window_end_sec` — annotated user turn end (same value)
- Harness streams the full user turn, then drains until trailing silence after
  model speech (or a no-speech timeout if the model never starts)
- Duplex models are drained by streaming frame-aligned silence, so they keep
  being stepped while they answer; hosted models get `end_input()` and a
  wall-clock wait. Both stop on the same speech definition.
- Score compares post-EOT model speech to the human assistant's next turn:
  floor time (first→last speech vs labeled `reference_duration_sec`),
  Whisper word counts, and response similarity (token F1 + embedding cosine
  on the same Whisper pair). Speech before the user EOT is ignored.
  `latency_ms` is the wait from that EOT to the onset of the first Whisper
  word in the model's post-EOT audio, mapped onto the playback clock — the
  same word evidence every other metric scores on. It is not the first
  above-gate interval (a model that murmurs into its first word puts both in
  one interval) and not the protocol text stream (which says the model
  produced text somewhere, not when). Above-gate audio that transcribes to no
  words is murmur, not an onset.

EOT points without a following assistant turn are skipped at prepare time.

## Timing for recall points

Relative to the live probe start (`rollout_user_input.wav` start):

- `checkpoint_sec` / `window_end_sec` — end of the synthesized probe (same value)
- History is a conversation prefix of 30 / 60 / 90 / 120 s, cut at the last
  completed turn at or before that target. `history_target_sec` is the IV;
  `history_sec` is the actual cut.
- Live protocol is the EOT drain. `speaker_2` asks; `speaker_1` is the model.
- Conversation-recall asks about a fact from the first 30 s (same question at
 every length). The extracted fact must pass an LLM judge for specificity and
 memorability, and a probe whose styled text leaks a literal speaker label is
 dropped. Fact-recall asks one item from a fixed list, assigned by
 `source_id`.
- Score is 1 if an LLM judge finds the gold fact in the post-checkpoint Whisper
  transcript, else 0. Timeouts still score whatever was said.

Recall points do not write interact `user.wav` / `assistant.wav` / `context.json`.
The probe WAV is synthesized once per source × metric at prepare time (Qwen3-TTS
clone from 20–40 s of `speaker_2`); eval replays that same file.

The same `status` rule applies: a rollout with no model audio is
`no_model_audio` and lands in `unscored`, not in the means as a zero
(or a zero-length / zero-similarity answer).

Both metrics measure speech with one shared definition — `speech_intervals` in
`bench/speech.py`, a hysteresis gate that opens at `rms_threshold` and holds to
half of it. The drain uses it to decide the model has stopped, and scoring uses
it to decide where speech was, so a rollout cannot be cut off at a point
scoring would still call speech.
## Speech gate

Pause drain and pause/EOT scoring share one hysteresis gate in `bench/speech.py`:
open at `rms_threshold` 0.01, hold to half of that, drop intervals shorter than
0.08 s. All three parameters are recorded in pause `score.json`. Do not retune
them off an RMS histogram; the calibration and why 0.01 stayed are in
`docs/phonation.md`.

`MAX_WORDLESS_SPEECH_RATIO` (0.05) in `bench/metrics/pause.py` is a triage
flag for above-gate audio while the model's text stream stayed empty. It is not
a claim about human listening rates.

## Pause scores are not turn-taking quality

A pause score asks whether the model uttered words that overlap the hold.
Quiet passes; above-gate murmur with an empty text stream also passes and is
counted in `wordless_phonation`. Do not quote a mean pause score without that
count.

PersonaPlex also has a bounded post-prefill warm-up (median ~7 s to first text
on full histories). Details, what was ruled out, and how to reproduce the
numbers: `docs/phonation.md` and `data/analysis/`.

## Output artifacts (per rollout)

Written under
`data/results/<run_id>/<eval_type>/<model>/<point_id>/rollout_NNN/`:

| File | What it is |
|------|------------|
| `trace.json` | Timed events (input frames sent, model audio received, signals, …). |
| `score.json` | Pause, response-length, or recall score, plus `"model"` so figures can be rebuilt without re-running eval. Respond-drain scores also store `latency_ms` (EOT to first word) and `wordless_phonation`. |
| `similarity.json` | EOT only: token F1 + embedding cosine vs the human reference, plus both texts. |
| `error.json` | Written instead of `score.json` when the rollout raises. |
| `response.wav` | Concatenated model output audio only. |
| `rollout.wav` | **Listening mix, not model input.** Stereo: left = streamed live user input (pause: pause_end plus drain silence for both families; EOT: the user turn), right = model output placed at wall-clock playback times. |

The run directory (`data/results/<run_id>/`) writes `run.json` (manifest /
point identity), a top-level `results.json` across eval types, plus
`pause.png` / `length.png` / `length-average.png` / `recall.png` /
`similarity.png` / `latency.png` when that eval produced scores, plus `index.html` linking those
root figures except `latency.png`. Each eval-type folder that ran also gets its own `results.json`.
Per-model figures live next to that model's rollouts:

```
data/results/<run_id>/
  run.json
  results.json
  index.html
  pause.png
  length.png
  length-average.png
  recall.png
  similarity.png
  latency.png
  pause-recognition/
    results.json
    gpt-realtime-fast/pause.png
    gpt-realtime-fast/<point_id>/rollout_NNN/
  response-length/
    results.json
    gpt-realtime-fast/length.png
    gpt-realtime-fast/similarity.png
    gpt-realtime-fast/<point_id>/rollout_NNN/
      score.json
      similarity.json
  conversation-recall/
    results.json
    gpt-realtime-fast/recall.png
    gpt-realtime-fast/<point_id>/rollout_NNN/
  fact-recall/
    results.json
    gpt-realtime-fast/recall.png
    gpt-realtime-fast/<point_id>/rollout_NNN/
```

Eval-type `results.json` is that eval's metrics flattened with `"eval"`:

```json
{
  "eval": "pause-recognition",
  "models": ["gpt-realtime-fast"],
  "manifest": "data/processed/example_1/manifest.json",
  "point": null,
  "rollout_count": 3,
  "mean_score": 1.0,
  "wordless_phonation": 0,
  "unscored": [],
  "errors": []
}
```

Run-level `results.json` is rebuilt from `score.json` / `error.json` on disk
(`"models"` lists every model under the run):

```json
{
  "models": ["gpt-realtime-fast"],
  "manifest": "data/processed/example_1/manifest.json",
  "point": null,
  "pause-recognition": {
    "rollout_count": 3,
    "mean_score": 1.0,
    "wordless_phonation": 0,
    "unscored": [],
    "errors": []
  },
  "response-length": {
    "rollout_count": 3,
    "mean_duration_ratio": 1.1,
    "mean_word_ratio": 0.9,
    "mean_duration_delta_sec": 0.2,
    "mean_word_delta": -1.0,
    "unscored": [],
    "errors": [],
    "timed_out": []
  },
  "conversation-recall": {
    "rollout_count": 8,
    "mean_score": 0.75,
    "by_history_sec": {"30": 1.0, "60": 1.0, "90": 0.5, "120": 0.5},
    "unscored": [],
    "errors": []
  },
  "fact-recall": {
    "rollout_count": 8,
    "mean_score": 1.0,
    "by_history_sec": {"30": 1.0, "60": 1.0, "90": 1.0, "120": 1.0},
    "unscored": [],
    "errors": []
  },
  "response-similarity": {
    "rollout_count": 3,
    "mean_token_f1": 0.3,
    "mean_embedding_cosine": 0.6,
    "unscored": [],
    "errors": [],
    "timed_out": []
  },
  "response-latency": {
    "rollout_count": 3,
    "by_eval": {
      "response-length": {
        "rollout_count": 3,
        "mean_latency_ms": 220.0,
        "median_latency_ms": 200.0,
        "p90_latency_ms": 310.0
      }
    },
    "wordless_phonation": 0,
    "unscored": [],
    "errors": []
  }
}
```

`bench eval` mirrors the run tree to
`s3://pmt-data-seamless/results/<run_id>/` with the local layout: each rollout
directory uploads when it finishes, run-level summaries when the run ends
(`bench/results_s3.py`).

Full runs execute on Modal, not the local machine
(`uv run modal run --detach bench/eval_modal.py --run-id <id>`): one container
per provider shard downloads the manifest's points from S3, restores any
mirrored `score.json` files so a restart never redoes scored work, and mirrors
each rollout as it finishes. The agent should keep watching the Modal logs the
way it would tail local ones — `uv run modal app logs people-bench-eval` —
to make sure the run is going smoothly (scored counts climbing, no provider
error bursts). After the run, `uv run python scripts/sync_results.py <run_id>`
pulls the mirrored tree into `data/results/<run_id>/` and rebuilds summaries;
per-shard run-level `results.json` on S3 only covers that shard's models, so
the synced local rebuild is the authoritative summary.

`--output` reuse scans the tree and merges: a second model into the same run
keeps the first model's scores. `uv run bench report data/results/<run_id>`
rebuilds summaries and PNGs from the stored scores without re-running eval.
Latency is a field on respond-drain `score.json`, not a third eval identity,
and is not recomputed from `trace.json`.

Each model sets `[model] max_concurrency` in `model.toml` (hosted APIs
are 8; PersonaPlex and Moshi are 64). `bench eval --model a b c` overlaps those
models; each keeps its own semaphore. PersonaPlex and Moshi are one A10G
container per session. `bench eval` accepts several `--manifest` paths in one
invocation and warms duplex GPU containers before the first rollout.

`rollout_count` counts scored rollouts only. A rollout that raised is recorded
in that metric's `errors` and the batch continues; a rollout with no model audio
is `unscored`. `wordless_phonation` counts scored pause rollouts that were above
the gate with an empty text stream. Response latency uses the same rule:
above-gate murmur with no words is not an onset, and is counted under
`response-latency.wordless_phonation` rather than the per-eval means.

A response-length or response-similarity rollout that hit `EOT_HARD_CAP_SEC`
stays in the means: a model that will not stop is a finding, not missing data.
Its duration is censored at the cap — a lower bound on what the model would
have said — while latency and similarity are computed on the words that exist
at the cut and are unaffected. The metric's `timed_out` lists those rollouts so
the censoring rate travels with the means; do not quote a duration mean for a
model with a high `timed_out` count without saying it is censored.

Root `pause.png` is a 100% stacked bar per model (Takeover / Hold / Wordless
/ No audio). `length.png` is a grid, one row per model. `length-average.png` is
the same two axes with one mean point per model. `recall.png` is
accuracy vs history length for conversation-recall and fact-recall.
`similarity.png` is token F1 vs embedding cosine for one model, or paired
strips when several models are in the run; point size is reference word
count. Per-model similarity figures live next to `length.png` under
`response-length/<model>/`. `latency.png` is a strip of EOT-to-first-word
latency per model, colored by model. Means stay in
`by_eval` so a conversation continuation is not averaged with a recall
probe. It is written next to the other root figures but not linked from
`index.html`. `uv run bench report data/results/<run_id>` regenerates them from
the stored scores.

### Why `rollout.wav` length can differ across models

Length is `max(live_user_stream, last_model_audio_end)`:

- Live user stream length is the streamed input only. For pause points both
  families stream the pause and the drain; for EOT the hosted stream ends at
  the user turn while duplex includes the streamed drain.
- If the model stays quiet, `rollout.wav` ≈ that user-stream length (silence
  after stream is wall-clock wait, not left-channel audio).
- If the model speaks after the pause window, the right channel extends the
  file. That is expected and does **not** by itself fail the pause score.

History audio (`rollout_user_context` / `rollout_assistant` / text context) is
**not** included in `rollout.wav`. Compare history via the point directory files
or preload logs, not via `rollout.wav` duration.

## Verifying the audio pipeline without listening

Everything here runs locally: no GPU, no Modal deploy, no model weights. Run it
before and after touching anything on the audio path.

```bash
uv sync --extra dev
uv run pytest tests/
```

| Suite | What it proves | A failure means |
|-------|----------------|-----------------|
| `test_frame_accumulator.py` | `bench.audio.FrameAccumulator` re-emits every sample in order for adversarial read sizes (480, 960, 2880, 1, prime, bursty, stalled). Each case also runs through `drop_remainder_frames`, the pre-fix loop, which must lose samples. | Either the accumulator drops audio, or the test no longer distinguishes it from the bug it exists to catch. |
| `test_personaplex_loopback.py` | The real `run_live_session` / `run_duplex_session`, over real websockets and the real sphn Opus codec, conserve user audio at every client chunk size, survive a model that blocks the event loop, and keep the model clock on wall clock. Also: the model is stepped through the whole scored pause window, the loop stays responsive during a long prefill, a prefill nothing reads the socket during still reaches `MSG_PRIMED`, an empty voice prompt clears the previous session's, and an idle session releases the model lock. Also: the assistant transcript survives JSON, the wire, and alignment as real word tokens at the right frames, rather than arriving as an all-PAD forced stream that a frame count alone cannot distinguish from a working one. | Audio is lost or reordered between `send_audio` and `lm_gen.step`, or a rollout is scored on a window the model never ran in. |
| `test_personaplex_adapter.py` | Client-side hostile-peer cases: an early message before PRIMED does not spin the prefill wait, a renamed sphn method raises instead of binding silence, the model's trailing sub-frame audio is flushed, and PCM16 round trips symmetrically. | The client can silently send or receive nothing while the trace still looks healthy. |
| `test_moshi.py` | Moshi is registered as a duplex model on the shared PersonaPlex adapter, the Modal serve command loads `kyutai/moshiko-pytorch-bf16` without PersonaPlex voices, and the same hostile-peer / flush / PCM16 cases hold when the adapter is loaded as `moshi`. | Moshi is missing from the registry, pointed at the wrong weights, or can silently drop audio. |
| `test_moshi_loopback.py` | The registered Moshi adapter, over the same loopback harness, conserves every sample and is stepped through the scored pause window. Sockets still open with `heartbeat=None`. | Adding Moshi forked the duplex path or the model is not stepped inside the hold. |
| `test_text_alignment.py` | The teacher-forced text stream is one sentencepiece token per frame, EPAD before each word run, no word truncated or overwritten. | Assistant history is primed with an out-of-distribution text stream. |
| `test_audio_metrics.py` | Calibrates every metric in `bench/audio_metrics.py` against signals of known character. | A metric moved, so thresholds that depend on it no longer mean what they say. |
| `test_rollout_analysis.py` | The artifact analyzer flags dropped frames, drones, stale scores, and audio above the speech gate from a model that emitted no text, on artifacts written by the real `run_rollout`. | Triage would miss a bad run. |
| `test_rollout.py` | `TraceSink` timestamps played audio on the same clock as `input_audio_sent`, still leaves real silence between frames, and a rollout whose transport died raises instead of returning a trace. | Playback timestamps drift against the input timeline, so every pause score is measured against a moving reference. |
| `test_pause_recognition.py` | A rollout with no model audio is `no_model_audio`, not a pass; decoded silence still scores; wordless audio during the hold is not a takeover; the speech gate survives an envelope dip mid-word; speech carried in from before the checkpoint is flagged separately; the gate opens on the block speech starts in, and all three of its calibrated parameters reach `score.json`. | Broken rollouts are scoring as good turn-taking, or murmur is counting as a response. |
| `test_response_length.py` | EOT response length is measured from the same hysteresis speech gate as pause scoring, ignores speech before the user EOT, a rollout with no model audio is `no_model_audio` rather than a zero-length answer, and one cut off by the drain cap is marked `timed_out`. | The EOT metric is measuring something other than what scoring calls speech, or a dead or censored rollout is being averaged in as a real response length. |
| `test_recall.py` | Prefix cuts land on completed turns, one probe is synthesized per metric and reused across history lengths, recall scores 1/0, a rollout with no model audio is `unscored`, and both live protocols turn VAD off the way they do for EOT. | History length is no longer the only IV, or a dead rollout is being counted as a missed fact. |
| `test_response_similarity.py` | EOT content scores token F1 and embedding cosine on the Whisper pair, empty text is zero not a match, a rollout with no model audio is `no_model_audio`, and one cut off by the drain cap is marked `timed_out`. | The content metric is scoring something other than the transcribed post-EOT reply against the stored reference. |
| `test_eval_seed.py` | Every rollout gets a distinct, reproducible seed. | Rollouts stopped being reproducible. |

`open_session_socket` must pass `heartbeat=None`, and both routes build their
socket through it so there is one place to get that wrong. aiohttp only observes
a pong from inside `ws.receive()`, and nothing reads the socket during duplex
prefill, so any heartbeat closes every history whose prefill outruns the pong
deadline however promptly the client answers: on the live server every point
under 18.5 s primed and every point over 33.1 s died with `ConnectionClosedOK`.
The peer that vanishes without a close frame, which is what a heartbeat would be
for, is covered by the idle watchdog instead.

### Loopback harness

`tests/personaplex_harness.py` drives the real server coroutines against a fake
Mimi/LMGen pair that echoes user audio back one frame later. `FakeServerState`
records every frame the server consumed, every seed and prompt it was handed,
and every teacher-forced text token, so tests assert on server-side state rather
than on decoded audio. Feed it `frame_ladder(n)`: each 80 ms frame is a tone one
FFT bin above the previous one and starts and ends at zero phase, so a lost
frame stays locatable after a lossy Opus round trip.

### Asserting on audio

`bench/audio_metrics.py` is the replacement for listening to a WAV.

| Metric | Reads |
|--------|-------|
| `rms`, `rms_envelope`, `speech_ratio`, `speech_intervals` | level and where speech sits, at the same 0.01 RMS threshold pause scoring uses |
| `spectral_flatness` | near 0 tonal, above 0.3 broadband noise |
| `spectral_flux` | below 0.05 spectrally frozen (a drone), above 0.2 moving like speech |
| `discontinuity_ratio` | largest sample step over peak; splicing a frame out pushes it well past 0.1 |
| `best_lag_correlation`, `envelope_similarity` | how closely what came back matches what was sent, and at what delay |
| `describe` | all of the above as one `AudioReport` |

### Triaging a real run

```bash
uv run bench analyze data/results/<run_id>
```

Reads `trace.json`, `score.json`, and `response.wav` per rollout:

| Signal | Meaning |
|--------|---------|
| `out/in` | model audio duration over streamed input duration. A duplex model steps once per received frame, so below 0.9 means frames were dropped or the server fell behind wall clock. This is the production symptom of a broken frame accumulator. The 0.9 threshold is calibrated for pause rollouts, where the streamed input has a fixed length. On EOT rollouts the drain keeps streaming for `EOT_TRAILING_SILENCE_SEC` after the model goes quiet, so that tail plus the few frames of pipeline latency is a constant subtraction — on a short rollout it alone can push the ratio under 0.9. Read the ratio against the rollout length before calling it dropped audio. |
| `lag` | worst gap between when an input frame was scheduled and when it was sent. Above one 80 ms frame, the source-time to wall-clock mapping that pause scoring depends on has drifted. |
| `flux` | spectral flux of `response.wav`. Loud but spectrally frozen output is flagged as a drone. Heuristic only, not yet calibrated against a known-good PersonaPlex rollout: treat it as a reason to look, not a verdict. This audio is *not* frozen — the wordless phonation runs 0.24-0.38 — so the drone gate does not catch it. |
| wordless phonation | share of `response.wav` above the speech gate while the model's text stream stayed empty. Above `MAX_WORDLESS_SPEECH_RATIO` (0.05) the model was voicing with no words behind it. A rollout can score 1 on pause recognition and still be 27% wordless phonation. Read it as exactly that and no more: the audio is a 125 Hz murmur 15 dB below the model's own speech, a person who listened called it acceptable, and the human rate the 0.05 was set against comes from a gated channel — see `docs/phonation.md`. |
| `status` | `no_model_audio` means nothing came back at all. Always a bug in the run, never a result. |
| score cross-check | `score.json` is rescored from the trace; a mismatch means the file is stale. |

A rollout can score 1 and still be flagged. Scoring only asks whether the model
stayed quiet through the pause; these signals ask whether the rollout was valid
at all.

### What still needs a GPU

The harness replaces Mimi and LMGen, so it says nothing about model quality,
prefill conditioning, or voice prompts. Those need
`uv run modal run models/personaplex/deploy/experiment.py` (teacher-forcing smoke test), `rank_probe.py` / `dose.py` (prefill rank and masking experiments), or a real `bench eval` against the deployed server. `realtime_factor` in the primed
payload is the number that says whether prefill keeps up on the current GPU. The
speech gate has now been calibrated (see `docs/phonation.md`), but only
against the artifacts of one deployed model, so re-check it on a run whose
responses are mostly lexical before trusting it on a different model.
