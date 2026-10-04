# PersonaPlex warm-up investigation

Lab notes from the pmt-2 duplex-path work. Operational contract (what each WAV is, who consumes it, how to test) lives in `AGENTS.md`. Reproduce the numbers cited here with `uv run python data/analysis/phonation/experiments.py`, `experiment_cohorts.py`, `prefill_gaps.py`, and `experiment_plots.py`; `rebuild_cache.py` regenerates the whisper decode cache if it is missing.

## Calibrating the speech gate

The gate has three parameters, and every one of them moves the pause score:
`rms_threshold` (0.01), `release_ratio` (0.5) and `min_duration_sec` (0.08).
All three are recorded in each pause `score.json`, so a score can be reproduced
from its own artifact.

They were calibrated against Whisper word timestamps over the model's own
`response.wav`, which is a label source independent of any RMS threshold. Each
response is tiled with 2 s windows at a 1 s hop and each window is peak
normalized before decoding, so the labeler cannot inherit the level bias being
measured; a word counts as a label only if whisper's own no-speech head,
mean logprob and repetition guard all pass, the word has non-zero duration, and
every window covering that instant agrees. On human reference speech the labeler
validates 49 of 62 windows (the misses are real pauses); on digital silence,
floor noise, broadband noise at 0.05 and a 120 Hz tone at 0.1 it labels nothing,
which is what makes it usable against whisper's habit of hallucinating
`" I'm sorry."` over silence.

Do not pick these values off the RMS histogram. An Otsu split of the same
21,580 envelope windows lands at 3.2e-4, and against the labels that threshold
opens intervals a median 840 ms **early** and drops the pause score from 0.544
to 0.228. Onset error keeps shrinking above 0.01 (MAE 282 ms → 169 ms at 0.05)
but only by dropping labeled speech: 0.01 is the highest swept threshold that
still matches every labeled utterance, and it is where the onset bias crosses
zero (median 0 ms on pause rollouts, -140 ms on EOT). A missed utterance is the
error that matters, because it turns a real takeover into a false pass. The same
rule picks `release_ratio`: 0.75 gives a better onset MAE (308 ms against
410 ms) but loses a labeled utterance, and `min_duration_sec` between 0.04 and
0.08 is a wash. So all three values are unchanged and scores from before this
calibration remain comparable with scores after it.

What the labels also show is that this metric is threshold-dominated and cannot
be fixed by moving the threshold. Block RMS is only a mediocre classifier for
lexical speech (AUC 0.80 on the pause run, 0.85 on EOT): labeled speech runs
down to 8e-4 while unlabeled audio runs up to 1.8e-2. On `baseline_v2`, 37 of 57
rollouts contain blocks above 0.01 with no labeled word anywhere in the file —
speech-like phonation (dominant frequency 110-135 Hz, spectral flux 0.24-0.38)
that whisper cannot decode, and that is not an echo of the user channel
(cross-correlation ≈ 0, anti-correlated envelopes). Lexical speech overlaps the
pause window in **1 of 57** rollouts, so the pause score of 0.544 is largely
measuring whether the model made a sound, not whether it took the floor.

## Reading this investigation cold

The sections below were written as the investigation went and two of their
leading theories were overturned by later measurements. This is the map. Every
claim here is tagged by how it is known, because the ones that read most
confidently are not always the best supported.

**Start with the listening report, because it reframes the rest.** A person has
now played `eot_baseline_10` and found it sensible: soft noise while the other
channel talks, then the assistant speaking normally at the turn end. The audio
the detector flags in `baseline_v2` is the **same sound** by every measurement
that separates noise from voicing, and the human control that made it a defect
is an artifact of the corpus recording chain. See "The listener heard it" below
before quoting anything from the defect section. What survives is the warm-up
(the text head takes seconds to engage) and its dose-response; what does not
survive is calling the sound itself a defect on this evidence.

**The harness bugs are fixed and are not the story.** Duplex rollouts used to
sleep through the pause instead of streaming it, so the model was never stepped
inside the window it was scored on; the frame accumulator dropped audio at
adversarial read sizes; a websocket heartbeat killed every history whose prefill
outran the pong deadline. All fixed, all covered by `tests/`, and pause scores
from before those fixes are not comparable with scores after them.

**Measured.** PersonaPlex phonates above the speech gate with an empty text
stream, at 0.909 of its above-gate audio on `baseline_v2` against 0.272 for GPT
Realtime on the same points and ~0.5 for human speech, which is the detector's
floor inside real speech (`experiment_cohorts.py`, `verdict.py`). It is a
**bounded warm-up**: raise the drain from 3 s to 15 s and 54 of 57 rollouts
produce text, median 7.2 s, p90 15.5 s (`experiments.py`). **Prefill length sets
its length**: 39 s of history gives 7.2 s, 4 s gives 4.6 s, one 80 ms frame gives
0.24 s with the wordless share falling to 0.311 (same script). Audio before the
first text token is 0.98-0.99 wordless in every condition; after it, 0.54-0.60.
Removing the forced text entirely (`exp2_padtext`) makes every measure **worse**.
And masking the voiced history frames that carry no word shortens the warm-up by
a restricted mean 1.91 s (CI -3.65 to -0.47) while removing the same amount of
voiced audio from worded frames does nothing (`dose_analysis.py`).

**Ruled out, and by what.** The teacher-forced text stream is correct — audited
token by token against `NVIDIA/personaplex` at the pinned `3428dfd` with the real
tokenizer (`alignment_audit.py`). The text head is not driven out of distribution
by prefill — median forced-token rank 0, p90 0-10 by token class, and the rank
*falls* across prefill rather than climbing (`rank_probe_analysis.py`). The empty
voice prompt, the last departure from upstream inference, changes none of it. The
plain PAD share of the forced stream predicts nothing (-0.00). And the voiced
frames with no word over them are **not** speech the transcript missed: 10.8% of
them carry a word an independent labeler finds, and the runs are median 100 ms
and at most 720 ms (`transcript_coverage.py`). And the phonation is **not white
noise and not a defect proven by the human control**: it is voiced at 125 Hz with
an 8.9 dB harmonic-to-noise ratio, but it sits within a decibel of the gate and
15 dB below the model's own speech, and the human channel it was compared against
is suppressed by 16.3 dB while the other speaker talks (measured below).

**Inferred, not established.** That the mechanism is the *pairing* of voiced
assistant audio with an empty text channel. The causal arm is consistent with it
and its confound control is flat, but it rests on **seven distinct duplex
histories** — the 19 pause points of `example_1` share only seven history files —
with a randomization p of 0.062, and the correlational version is p 0.048 on the
same seven. Treat it as supported and unfinished. Nothing here explains *why*
that pairing would matter given that the same micro-gap structure must have been
in training.

**Open, and what a person should do next.** (1) Replicate
the dose arm on histories from other conversations; seven is the binding limit on
everything above and more points from `example_1` will not help, because they
reuse the same histories. (2) The pause metric is not a measure of turn-taking
quality on its own — see the section of that name — and making it diagnostic
needs a new point type, which is a prepare-time design job. (3) A human control
worth the name needs a corpus whose listening channel is not gated; the one on
disk cannot supply it.

**Reproducing without a GPU.** The scripts under `data/analysis/` are in the
repo; the artifacts they read (`data/processed`, `data/results`, the whisper
decode cache `windows.json`) are not, since `/data` is otherwise ignored. On a
machine with the runs on disk every number above re-derives from those scripts.
`rebuild_cache.py` regenerates the decode cache if it is missing, and it is slow
rather than difficult.

## The wordless phonation is not backchannelling

**The "defect" half of this section is superseded by "The listener heard it"
below.** What holds up is everything about *what the audio is*: it is not
transcribable speech, not a "mm-hm", and not an artifact of the scorer. What does
not hold up is the verdict, because the human control below rests on a corpus
channel that is gated, and a person has since listened and disagreed.

The obvious benign reading of that phonation is that it is backchannelling —
"mm-hm", breaths, filled pauses — which whisper does not transcribe and which
would mean the audio path is fine and the scorer needs a notion of "a
backchannel is not taking the floor". It is not. Four independent measurements
say so, and all of them are reproducible from artifacts already on disk.

**The model's own text stream.** Every adapter reports it and `trace.json`
records it, so it is a word label that owes nothing to whisper or to any RMS
threshold. In **30 of 57** `baseline_v2` pause rollouts PersonaPlex emitted *zero*
text tokens, and those 30 rollouts still contain **85.9 s** of audio above the
0.01 gate, a median 34% of the rollout, with zero lexical seconds. Whisper and
the text stream never disagree in the direction that would matter: across every
run on disk there is **no** rollout where whisper found a word the model did not
claim to say. Whisper is conservative here, not wrong.

**The human control, co-timed.** For each pause point, `assistant.wav` over
`[history_end_sec, input_end_sec]` is the real other speaker across exactly the
window the model is scored on. Over 19 points and 92 s, that channel **never
reaches the gate** — peak cell RMS 0.0075 against a 0.01 threshold. It is not a
hard diarization mask either: the channel carries a real noise floor and 38.8 s
of above-gate audio elsewhere. Widening to the whole conversation, each speaker
puts 0.08-0.12 s above the gate while the other holds the floor, 0.8% of 25 s of
listening. PersonaPlex is above the gate for 34.6% of every pause rollout.
**This control does not survive the gated-channel measurement below.** "Not a
hard diarization mask" was measured as "the channel carries a noise floor", which
it does, but it is also 10.8% literally muted with a 0.727 s
run of exact digital zeros, and its floor drops 16.3 dB while the other speaker
talks. A channel cannot get quieter because someone else started talking.

**Levels are comparable across cohorts, which is not the same as saying the
flagged audio is loud.** Median RMS of lexical cells: model pause 2.8e-2, model
EOT 3.0e-2, human channel 4.2e-2, human reply turn 5.0e-2. Rescaling every cohort
to a common anchor moves the model's wordless share of above-gate audio from 0.909
to 0.912. That makes the absolute gate fair between cohorts. Within the model,
though, the wordless audio is 15 dB below its own speech — see below.

**Acoustics.** ~0.5 is the detector's floor inside genuine speech, because
whisper word timestamps do not tile an utterance: human audio sits at 0.49-0.51
and the model's own EOT output at 0.58. Pause rollouts sit at **0.909**. Wordless
spans reach 4.36 s where human wordless spans never exceed 1.46 s, and 45% of
them carry a single LPC resonance below 4 kHz and 45% carry none, against two for
the human ones — voiced, harmonic, ~125 Hz, and not shaped like a vowel. Decoding
those spans alone, peak normalized, forced to English and primed with
backchannel tokens returns hallucinations (`"I'm sorry."`, a repetition loop) that
fail every confidence gate; it never returns "mm-hm". Only whisper `base` is
cached locally, so a larger model has not been tried.

**A different model under the identical protocol does not do it.** GPT Realtime
over the same 19 pause points leaves 0.27 of its above-gate audio wordless and
emitted text in every rollout, so neither the protocol nor the scorer produces
this.

It also predates every fix and is not the cleared voice prompt. The pre-fix
`baseline` run is 0.973 wordless with 14 of 21 rollouts textless, and the
`voice_prompt='NATF2.pt'` run is 0.860 with 30 of 37 textless. Over the
pre-checkpoint region, which both protocol versions streamed identically as real
frames, clearing the voice prompt took above-gate audio from 10.2% to 16.7% on 36
matched rollouts while more than doubling the share of rollouts producing real
words (19% to 47%): the model became more vocally active in both directions. The
phonation rate correlates with rollout length (rho +0.46) and not with position in
the run (rho -0.002), and it is spread throughout rather than clustered at the
pause: 56.9 s before the checkpoint, 12.9 s inside the scored window, 65.9 s
after.

**Why EOT transcribes and pause does not** is the same phonation seen for longer.
PersonaPlex vocalizes above the gate for seconds before its first text token in
*both* protocols — on EOT point 000013, phonation starts at 0.70 s and the first
text token lands at 13.78 s. EOT rollouts run a median 14.6 s and end only after
real speech, so words dominate the file; pause rollouts run a median 6.9 s, and
the 27 that did produce text did so at a median 4.9 s with 1.8 s of rollout left.

So the scorer is not measuring nothing: audio above the gate with an empty text
stream is a real quantity, and nothing else surfaced it — `out/in` only measures
frame conservation and the drone gate wants flux below 0.05 where this audio sits
at 0.24-0.38. `bench analyze` now flags it, which takes `baseline_v2` from 0 of 57
rollouts flagged to 18 of 57. Read the flag as "this rollout made voiced sound it
had no words for", which is what it measures; the claim it originally carried,
that this is *failing turn-taking*, is the part the section below withdraws.

## The listener heard it, and it is a quiet murmur while the other party talks

A person has now played `eot_baseline_10` and reports that the rollouts "seem
pretty sensible", that the output is a "soft white noise type output" which "only
happens while someone is talking on the other channel", and that "when their turn
is up then the assistant speaks as expected". The timing claims are now measured
and confirmed exactly as stated; "white noise" is half right; and the consequence
is that the defect verdict above is stronger than the evidence.

`character.py` and `control.py` are the remaining reproduce path for the
acoustics and the gated human channel; the listening-clip generator was never
run, and a person already judged `eot_baseline_10` instead.

**Measured: the timing is exactly as reported.** Over the 10 `eot_baseline_10`
rollouts there are **0.00 s** of lexical audio before the streamed user turn ends
and **36.6 s** after it, and the first text token lands after the turn end in
**10 of 10**. Above the gate the model is at 36.7% before the boundary and 34.0%
after, so it is making sound throughout and the words are all on the far side.
"Noise while they talk, then speaking when their turn is up" is not an impression;
it is what the traces say.

**Measured: voiced, not noise — but quiet and unarticulated.** Medians over
121.0 s of flagged pause audio in 156 spans: harmonic-to-noise ratio **8.9 dB**,
overlap-normalized autocorrelation peak **0.89**, spectral flatness **0.000**,
zero crossing rate **283 Hz**, f0 **125 Hz**, voiced fraction **0.98**. Measured
by the same code, white noise at the same level reads HNR **-9.3 dB**, flatness
**0.561**, ZCR **12006 Hz**, and a 125 Hz vowel reads HNR 60 dB (r = 1.00),
flatness 0.000, ZCR 250 Hz. So literally it is periodic, not aperiodic, and
"moaning" is closer than "white noise" on that axis alone. On every other axis
the listener's word is the better one: the flagged audio sits at RMS **0.0108**,
**+0.7 dB** relative to the 0.01 gate and **-15.4 dB** relative to human speech,
with a spectral centroid of **139 Hz** and **0.000** of its energy above 1 kHz,
against 394-408 Hz and 0.016-0.020 for real speech from either the model or a
human. A quiet, dull, 125 Hz murmur with nothing above 1 kHz has no consonants
and no articulation in it: it is a sound, not a voice, and at 15 dB down it is a
background one. That reconciles the report with the acoustics, and it is why
"moaning noises" set the wrong expectation for everyone who read it.

**Measured: the two cohorts are one phenomenon.** The flagged pause audio and the
EOT pre-first-text audio agree on every metric — RMS 0.011 against 0.010, HNR 8.9
against 8.2, ZCR 283 against 290, centroid 139 against 142, f0 125 against 123,
voiced 0.98 against 0.96 — and the probability that a random pause segment exceeds
a random EOT segment stays between **0.40 and 0.63** on all ten, with 0.50 on ZCR.
There is no second phenomenon to find. **The audio a person has already heard and
accepted is the audio the 0.909 wordless share is counting**, and that is the
single most important thing in this document that a metric did not produce.

**Measured: the human control is an artifact of the corpus recording chain.** The
corpus is two per-speaker channels (`manifest.source_audio` is `channel_1.wav` and
`channel_2.wav`) and `prepare_eval` only slices them, so anything in them came
from how the conversation was recorded. What is in them: **26.1%** of the
assistant channel's samples are exactly zero, **10.8%** of its 20 ms cells are
90%+ zeros, and it holds one **0.727 s** run of contiguous digital zeros with 19
runs over 20 ms. In the 19 co-timed listening windows, 10-53% of samples are exact
zeros, up to 34% of cells are muted, and in half of them the quietest decile is
85-99% zeros carrying **3-5 distinct sample values**. Speech onsets climb from
floor to half peak inside a single 20 ms cell in **85-92%** of cases. And the
decisive one: the assistant channel's floor while the *other* speaker talks is
4.9e-5 against 3.2e-4 when both are quiet, **-16.3 dB** (the user channel,
-5.3 dB). A live microphone cannot get 16 dB quieter because someone else started
speaking. This channel is gated — platform-side suppression or DTX, upstream of
this repo — so "the human never reaches the gate while listening" is a fact about
the recording and not about human conversation. It cannot carry a verdict on the
model, and the 0.8%-of-25 s figure the `MAX_WORDLESS_SPEECH_RATIO` comment quotes
is a lower bound on human listening noise, not a measurement of it.

**Measured: the gate is close to, but not only, a floor detector.** Raising the
speech gate from 0.01 takes the pause cohort's flagged seconds 135.6 → 92.3 → 70.6
→ 41.6 → 16.1 at 0.015, 0.02, 0.03 and 0.05, so the bulk sits within 6 dB of the
threshold, and yet 70.6 s survives at double the gate and 16.1 s at five times it.
Half of the flagged audio would be gone at a threshold that still misses **0 of
35** labeled pause utterances and **0 of 49** human ones. This is not a reason to
move the gate — the calibration in "Calibrating the speech gate" turns on onset
timing and on missing no utterance, and both still favour 0.01 — but it does mean
that a wordless-phonation share is reported against a threshold most of this audio
only just clears.

**Measured: `exp2_hist0`'s remaining wordless share is a different thing.** Its
flagged spans sit at RMS 0.0539 (**+14.6 dB** re gate), centroid 360 Hz, HNR
11.4 dB — indistinguishable from real speech on every axis, and 15 dB above the
pause cohort. So "the drone disappears when the history does" is right about the
murmur and wrong in tone: what is left at 0.311 is the detector's ordinary floor
inside audible talking, not a quieter version of the same defect.

**Reassessed: on this evidence, soft noise while listening is behaviour, not a
defect.** The three legs of the defect verdict were the human control (now an
artifact), the level comparison (comparable across cohorts, but the flagged audio
is 15 dB below the model's own speech) and the acoustics (voiced and
unarticulated, which is what a listening noise would be). The cross-model control
survives — GPT Realtime leaves 0.27 of its above-gate audio wordless against
PersonaPlex's 0.909 — so PersonaPlex does make more of this sound than a hosted
model does, and the warm-up and its dose-response are unaffected by any of this.
What is no longer supported is "a model that phonates over the user is failing
turn-taking". A duplex model that murmurs 15 dB below its speaking level while the
other party talks and then answers at the turn end is doing something a listener
recognizes as normal. Two things would still change that reading, and both are
listening questions rather than measurement ones: the loud tail (the worst flagged
span reaches RMS 0.114, **louder** than the model's own answering voice, in a
rollout that emitted no text at all) and the length (wordless spans reach 4.36 s
where human ones never exceed 1.46 s).

**A person already closed the listening question for EOT.** Pause-run clips
were never generated. The remaining listening question is whether the loud
flagged tail (the worst span reaches RMS 0.114, **louder** than the model's own
answering voice, in a rollout that emitted no text at all) and the 4.36 s
wordless spans sound like the quiet murmur a person accepted on
`eot_baseline_10`.

## The text conditioning is not the cause

The obvious suspect for an empty text stream beside continued audio is the
teacher-forced inner monologue, and `align_words_to_frames` had been rewritten
for structure without ever being checked against the model. It has now been
audited against `NVIDIA/personaplex` at the pinned `3428dfd`, and it is correct
on every axis that has an upstream reference. **Do not re-audit this path from
scratch; look at the voice prompt, the audio path, or the checkpoint instead.**

The special tokens are right. `moshi/moshi/models/lm.py` sets
`LMGen.zero_text_code = 3` and `LMModel.end_of_text_padding_id` returns `0`,
`loaders.py` sets `existing_text_padding_id: 3`, and both `server.py` and
`offline.py` name the low ids `['EPAD', 'BOS', 'EOS', 'PAD']`. The tokenizer
agrees: `tokenizer_spm_32k_3.model` has `<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3,
and is byte-identical (sha256 `78d43365…`) to kyutai's, so kyutai references
apply to it. So `PAD_TOKEN = 3` and `EPAD_TOKEN = 0` are both correct, and they
are not reversed.

The delay structure is right, and needs no offset here. `_lm_kwargs["delays"]` is
`[0, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1]`: text and both semantic codebooks are at
delay 0, and the one-frame acoustic delay applies only to codebooks 2-8 and
10-16. `LMGen.prepare_step_input` writes the forced text token at
`state.offset + delays[0]` itself, so the caller must hand it the text for the
*same* frame as the audio, which is what `run_duplex_prefill` does. Teacher
forcing also lands on the right streams: `step(input_tokens, moshi_tokens,
text_token)` routes `input_tokens` to the user codebooks (k=9-16) and
`moshi_tokens` to the agent's (k=1-8), and prefill passes user, assistant, text
in that order.

Assistant-only text with no BOS/EOS is right. moshi-finetune's `train.py` builds
its `Interleaver` with `keep_main_only=True`, which excludes `_insert_bos_eos`
entirely, and every run on disk corroborates it: PersonaPlex's own text stream
contains only its own speech and never transcribes the user. Word placement is
right too -- the repo admits a word at `floor(start * 12.5)` and the
interleaver's `start * rate < t + 1` selects the same frame. Prompt framing
matches `server.py` exactly (`wrap_with_system_tags`, then a raw `encode`, one
token per frame against silence and sine), at 110 tokens against upstream's own
73-token test asset.

One real divergence exists and is measured to be immaterial: the repo queues a
word's leftover pieces where the interleaver's default (`keep_and_shift=False`)
drops them when the next word starts. Over all 29 point histories the worst word
lands 0.16 s (two frames) late, 12 of 29 streams are byte-identical, the worst
differs in 22 of 745 frames, and the upstream default would discard 88 of 1672
pieces. NVIDIA published no training code at the pinned commit, so there is no
authoritative answer for *this* checkpoint, and changing conditioning on kyutai's
default alone would be a guess. Left as-is deliberately. The histories are also
not degenerately empty, which was the other live hypothesis: they run 77.8% to
92.3% PAD, about one word per second, which is the expected density for one
speaker in a two-party conversation. Rerun with
`data/analysis/text_conditioning/alignment_audit.py`, which writes
`alignment_audit.json` beside itself and needs a real sentencepiece model.

What the audit does *not* clear is the voice prompt. PersonaPlex `model.toml`
sets `voice_prompt = ""`, so `configure_session` calls `clear_voice_prompt()` and
`_step_voice_prompt_core` steps zero frames. Upstream has no such mode:
`server.py` raises `FileNotFoundError` when the voice prompt is missing and
`offline.py` makes `--voice-prompt` required, so every supported upstream path
conditions the agent stream on a voice prompt before anything else, and the `.pt`
path additionally restores a saved LM cache. Running with none is the largest
remaining departure from upstream inference on this path, and it is upstream of
the text head rather than in it.

That has now been measured on a GPU, and the answer is under "The forced text is
in distribution" below: the rank does not climb, and the voice prompt does not
change it.

## The silent text head is a bounded warm-up, and prefill length sets its length

Four live runs on a server deployed from the pinned `cdbbd95`, all 19
`example_1` pause points, all with the drain raised so a pause rollout lasts as
long as an EOT one (18.7-18.9 s of model audio observed on every rollout, so the
time-to-first-text distribution below is uncensored out to 15 s).

```bash
uv run bench eval personaplex data/processed/example_1/manifest.json \
  --metric pause-recognition --rollouts 3 --pause-drain-sec 15 --run-id exp1_drain15
```

`--pause-drain-sec` sets the post-`window_end` drain (default `PAUSE_DRAIN_SEC`,
3.0 s). History-length and all-PAD-text ablations are not on `bench eval`; they
go through `bench.history.condition_duplex_history` from
`models/personaplex/deploy/rank_probe.py`.

| run | history | forced text | n | any text | first text p25/p50/p75/p90 | wordless share | textless |
|-----|---------|-------------|---|----------|----------------------------|----------------|----------|
| `baseline_v2` (drain 3 s) | full | aligned | 57 | 27/57 | 3.3 / 4.9 / 6.8 / 8.0 | 0.909 | 30/57 |
| `exp1_drain15` | full | aligned | 57 | **54/57** | 4.6 / 7.2 / 10.8 / 15.5 | 0.755 | 3/57 |
| `exp2_padtext` | full | all PAD | 38 | 21/38 | 6.6 / 9.3 / 16.6 / 16.8 | 0.811 | 17/38 |
| `exp2_hist4` | last 4 s | aligned | 38 | 35/38 | 3.3 / 4.6 / 6.3 / 7.6 | 0.669 | 3/38 |
| `exp2_hist0` | one 80 ms frame | aligned | 38 | **36/38** | 0.1 / 0.2 / 3.8 / 5.1 | **0.311** | 2/38 |

`textless` counts rollouts with above-gate audio and no text token at all;
`wordless share` is `detect.py`'s phonation over above-gate audio, the same
number that reads 0.909 on `baseline_v2` and 0.272 on GPT Realtime.

**It is a warm-up, not an unbounded state.** Changing only the drain takes the
textless rate from 30 of 57 to 3 of 57. The 30 silent rollouts in `baseline_v2`
were an artifact of watching for a median 6.9 s: the same conditioning observed
for 18.7 s produces text in 95% of rollouts, at a median 7.2 s with a p90 of
15.5 s and a maximum of 23.4 s. The headline is that the text head needs seconds
to engage, and the phonation is what the audio head does in the meantime — the
audio before the first token is 0.98-0.99 wordless in every condition that has
any, and the audio after it is 0.54-0.60, which is the EOT cohort's 0.584 and
close to the detector's ~0.5 floor inside genuine speech. Nothing here says the
model is stuck; it says it starts wrong and recovers.

**Prefill length is what sets the warm-up.** Trimming history to the last 4 s
halves the median (7.2 s to 4.6 s), and cutting it to a single frame collapses it
to 0.24 s: with no conditioning the text head engages essentially immediately,
and the wordless share falls to 0.311, within a whisker of GPT Realtime's 0.272
on these same points. `exp2_hist0` produces **zero** above-gate audio before its
first text token. That is the cleanest statement available that the phonation is
prefill-induced rather than an audio-path or decoder defect: the same server, the
same speech gate, the same points, and the drone disappears when the history
does.

**Malformed teacher forcing is not the mechanism, and unworded prefill audio
is.** `exp2_padtext` is the audit's hypothesis run — same history audio, no word
stream — and it comes out *worse* than `exp1_drain15` on every axis (21 of 38
producing text against 54 of 57, median 9.3 s against 7.2 s, wordless share 0.811
against 0.755). Removing the forced text does not help, so "the forced text is
malformed" is contradicted, consistent with the upstream audio above. What
padtext is, though, is the maximum dose of voiced assistant audio that no word
accounts for, and that quantity predicts the warm-up across points:
spearman(unworded share of voiced prefill, median time-to-first-text) = **+0.67**
over the 19 points, against +0.44 for unworded seconds, +0.35 for history length
and +0.10 for voiced seconds. Aligned text is a weaker dose of the same thing —
whisper timestamps do not tile an utterance, so 33.5 s of the 176.7 s of voiced
history is PAD under audible speech. Read this as a lead rather than a result:
the 19 points share only seven distinct histories, and the ordering of padtext
against drain15 is the load-bearing part. The dose has since been intervened on
directly rather than correlated — see "Removing the dose does shorten the
warm-up" — and the intuitive account of *why* it would matter, that whisper
missed real speech at prepare time, has been measured and is false.

**None of this is visible in the pause score**, which moves 0.544 (baseline) →
0.526 (drain15) → 0.789 (padtext) → 0.526 (hist4) → 0.553 (hist0). The condition
that removes the phonation scores the same as the one that maximises it, and the
worst condition scores best, because a model that goes quiet passes. This is the
same "the score is measuring whether the model made a sound" result as above,
now demonstrated by moving the defect without moving the metric. `exp2_hist0` is
also the *most* vocally active condition over the first 3.7 s (29.3% above the
gate against baseline's 17.6%) — it is not quieter, it is talking.

All four runs are healthy: every rollout scored, none unscored, `out/in` at worst
0.92, worst lag 63 ms, no truncated history. A handful of rollouts died on local
DNS failures mid-run and were re-run per point, so `results.json` for
`exp2_padtext` and `exp2_hist4` describes only the patch batch; the analysis
scripts derive counts from per-rollout artifacts for that reason.

Reproduce, all from the artifacts on disk:
`data/analysis/phonation/experiments.py` (trace-only: health, the
time-to-first-text distribution, above-gate audio split at the first token),
`experiment_cohorts.py` (the whisper pass through `detect.py`, which extends
`cohorts.json` and leaves the old rows untouched), `prefill_gaps.py` (the
unworded-history dose-response) and `experiment_plots.py`, which writes
`experiments.png` and `experiments_mechanism.png`.

## The forced text is in distribution, and the voice prompt is not the cause

The remaining mechanism hypothesis was that prefill drives the text head out of
distribution, with the unvalidated empty voice prompt as the suspect upstream of
it. Both are now measured directly rather than inferred from outputs, and both
are **negative**.

```bash
uv run modal run models/personaplex/deploy/rank_probe.py::probe
```

That runs a voice prompt (`NATF2.pt` vs none) x history length (full vs last
4 s) factorial over three points spanning 12.5 s to 59.6 s of history, and
records, for every prefill frame, the rank the text head gives the token prefill
forces on it. Rank 0 means the model would have emitted that token itself. It
drives the real `ServerState` and the real `run_duplex_prefill`, so only the
ablation varies, then continues each arm on 15 s of silence frames the way the
drain does, which is what lets a rank trajectory be read against a
time-to-first-text. Analysis and figure:
`data/analysis/voice_prompt/rank_probe_analysis.py`, writing `rank_probe.png`.

**The forced text is what the model would have said anyway.** Over full
histories the median rank is 0 for every token class, the p90 is 0 for PAD, 1 for
EPAD and 8-10 for word pieces, and across up to 745 frames only **0 to 2 frames**
have a rank above 100 — frame 0, and one word onset. Median probability on the
forced token is 0.90-0.97. Teacher forcing is not fighting the model.

**The rank does not climb; if anything it falls.** Mean rank by fifth of prefill
on the 59.6 s point is 46.6, 0.6, 0.2, 0.3, 2.2: the only elevated stretch is the
first fifth, driven by the single frame where the model steps out of the system
prompt into forced history. And the 4 s arms score *worse* than the full-history
arms (mean rank 24.7-70.3 against 2.4-14.5), because less context makes the next
word less predictable. So the conditioning that produces the longest warm-up is
the conditioning the text head finds *most* predictable, which rules out
"out-of-distribution forced text" as the mechanism.

**The voice prompt changes none of it.** With full history the two voice arms
agree to three decimals on median rank, p90 by token class, and probability on
the forced token. On time-to-first-text in the continuation it is a wash at
n=3 points and not in the direction of restoring it: without a voice prompt
000006 reaches text at 1.6 s where the voice-prompt arm never does inside 15 s,
000029 at 0.3 s against 12.5 s, and 000016 is the one point that goes the other
way (2.4 s with, never without). Combined with the earlier run-level result that
clearing the voice prompt more than doubled the share of rollouts producing real
words (19% to 47%), there is no evidence that restoring it fixes anything, and
this closes the last unvalidated departure from upstream inference. **Do not
spend more GPU on the voice prompt.**

**What the probe supports is a prior the model holds confidently, not a
corruption.** Prefill teacher-forces PAD on 56-89% of frames, and the model
*agrees*: on PAD frames its own top-1 token is PAD in 86-100% of cases, and it
ends prefill assigning 0.81-0.95 to the token it is being forced to emit, which
by then is PAD in 83-100% of the last second. Nothing is broken in the text head;
it has been shown a stretch of its own speech and told, confidently and
correctly, that it is saying nothing.

**And it is specifically PAD under voiced audio, not PAD mass.** The obvious
reading of the paragraph above — that a mostly-silent text channel is the dose —
is wrong, and `prefill_gaps.py` tests it directly: the plain PAD share of the
forced stream correlates with median time-to-first-text at spearman **-0.00**,
and the PAD share of the last 2 s of prefill at **-0.04**, against **+0.67** for
the unworded share of *voiced* history. What co-varies with the warm-up is
history frames where the assistant channel is audibly speaking and the text
channel says nothing — 33.5 s of the 176.7 s of voiced history on these points,
because whisper word timestamps do not tile an utterance.

**The honest unit here is 7, not 19.** The 19 pause points of `example_1` carry
only **seven distinct duplex histories** (points from one conversation share the
history file byte for byte), and prefill is the whole of what this measurement
depends on, so the point count overstates the evidence. `prefill_gaps.py` now
reports both. Deduplicated, the dose correlation survives and strengthens
(**+0.79**, exact permutation p **0.048**, partialled for length **+0.80**) while
history length weakens to **+0.18**. That is one test on seven units at the edge
of significance, which is why the causal arm below matters more than the
coefficient.

**The dose is not speech the transcript missed.** The intuitive account of why
that quantity would matter is that whisper drops backchannels, partial words and
quiet speech at prepare time, so the model is primed with real speech labeled as
nothing — a labeling gap that training would not have contained, and a fix that
would belong in transcript coverage rather than in `text_alignment.py`. That
account is **measured and false**. `transcript_coverage.py` re-labels every
`rollout_assistant.wav` with the independent labeler in `detect.py` (2 s windows
at a 1 s hop, each peak normalized, a cell speech only when every window covering
it puts a word over it) and compares it against the stored transcript: of the
55.7 s of voiced assistant history no stored word covers, only **6.0 s (10.8%)**
carries a word the independent labeler can find, only **4.3 s (7.8%)** sits in
runs of 0.5 s or longer, and the 421 unworded runs are median **100 ms**, p90
**280 ms**, maximum **720 ms**. That is the gap structure every word-timestamp
stream has — plosive closures, inter-word transitions, the RMS gate's hangover
past a word's end — not dropped utterances. So the transcripts do **not**
systematically under-cover voiced assistant audio, the same micro-gap structure
would have been present in training, and the "prefill teaches a pairing the model
never saw" reading of the correlation loses its mechanism before any GPU is
spent.

Upstream's own instrument cannot be used for this on the pinned commit.
`LMGen(report_loss=True)` forces `return_logits` on and then `create_loss_report`
rebinds `target` to the text channel at `lm.py:601` before indexing
`target[:, k + 1]` at `lm.py:623`, which raises `IndexError` on the now 1-D tensor
for any `dep_q > 0`; PersonaPlex has `dep_q = 8`. The probe therefore takes the
same quantity off the text logits by wrapping the graphed forward. Two further
things this path needs and the deployed server gets for free: `torch.no_grad()`,
which `duplex_server.main` holds around everything and the mimi passes in
`prefill` inherit rather than declare, and keeping per-step allocations off the
device, because this model leaves tens of megabytes spare on an A10G and one
extra device allocation per frame fragments the pool into an OOM.

## Removing the dose does shorten the warm-up, and removing the same audio elsewhere does not

The causal arm of the above, run because a correlation over seven histories
cannot carry the claim on its own.

```bash
uv run modal run models/personaplex/deploy/dose.py::dose --seeds 5
```

Three prefill arms per distinct history, five seeds per cell, 105 runs. Prefill
is fully teacher-forced and therefore deterministic, so the seed only varies the
continuation, which is where the warm-up is measured; two arms with the same seed
and the same conditioning return the same frame.

- `control` — the unmasked history.
- `mask_unworded` — the voiced history frames no stored word covers, zeroed on
  the LM's own 80 ms grid. 0.08-3.28 s per history, 9.52 s over the seven.
- `mask_worded` — the same *number* of voiced frames, spread evenly over frames a
  word does cover. Removes the same amount of voiced assistant audio and chops
  the history the same way, so a positive result cannot be read as having simply
  fed the model less audio.

| outcome | `mask_unworded` vs control | `mask_worded` vs control |
|---------|----------------------------|--------------------------|
| time to first text, restricted mean | **-1.91 s** (95% CI -3.65 to -0.47) | -0.04 s (95% CI -2.12 to +1.93) |
| histories improved | **6 of 7**, randomization p **0.062** | 3 of 6, p 1.000 |
| produced any text inside 15 s | 30/35 against 23/35, **+0.20** (CI +0.06 to +0.37) | 23/35, +0.00 (CI -0.11 to +0.11) |
| share of continuation above the gate | +0.084 (CI +0.013 to +0.150) | +0.066 (CI -0.001 to +0.133) |

**The result is positive and specific, and it is at the edge of what seven
histories can resolve.** Read the interval rather than the p: the cluster
bootstrap excludes zero on both primary outcomes, and the exact randomization
test over all 128 sign flips of the seven paired histories gives 0.062 two-sided.
The estimator is a *restricted* mean because a third of runs never produce text
inside the 15 s continuation; censoring is identical across arms, so the
difference is honest but conservative, and a median of per-history medians
collapses to zero whenever both arms are censored. Uncertainty is a cluster
bootstrap over histories, because five seeds inside one history share one prefill
state and are repeated measurements of a condition rather than new draws of it.

**What makes it more than "less audio" is the control arm sitting at zero.**
Both masking arms remove 9.52 s of voiced assistant audio in ~100-300 ms pieces;
only the arm that removes the voiced-without-a-token frames moves the warm-up.
The control arm is not perfectly neutral — zeroing frames a word covers leaves
words with no audio under them, which is a different inconsistency rather than no
inconsistency — so if that were itself harmful it could hide a benefit. It shows
no benefit and mildly *more* pre-text phonation (+0.055), which points the same
way rather than the other.

**Masking makes the model more vocally active, not quieter.** Above-gate share
rises in both masking arms, and for `mask_unworded` the extra audio comes with
words: it reaches text more often and sooner while phonating before the first
token no more than control does (+0.007). This is the same shape as `exp2_hist0`,
which was the most vocally active condition and the least wordless.

**Masking does not push the forced text out of distribution.** All three arms
hold a median forced-token rank of 0 and a p90 of 1, so the intervention removed
audio without making the text stream harder to predict.

Together with the negative on transcript coverage this leaves a specific and
slightly awkward conclusion, which is stated rather than smoothed: voiced
assistant history frames that carry no text token are causally implicated in the
warm-up, **and they are not a labeling defect**. There is nothing to fix at
prepare time — those frames are the ordinary gaps between word timestamps. Any
intervention that removes them either densifies the forced text, which departs
from the alignment the upstream audit cleared, or silences real history audio,
which departs from the conversation the benchmark is supposed to present. So this
is a characterization of the checkpoint's prefill sensitivity, not a bug with a
patch: **do not** ship either masking arm as a benchmark default.

Reproduce with `data/analysis/dose/dose_analysis.py`, which reads
`dose_probe.json` and writes `dose.png`.

## Capping duplex history is a knob, and it changes what is measured

`bench.history.condition_duplex_history(history_sec=N)` primes duplex models with
only the most recent N seconds of history. The rank-probe entrypoint is the
place that uses it; it is not a `bench eval` flag and must not become a silent
default. Runs at different history lengths are not comparable.

The trade-off is real in both directions. Shorter history is the largest lever
on the defect measured so far — 39 s of history gives a median 7.2 s to first
text, 4 s gives 4.6 s, and one 80 ms frame gives 0.24 s with the wordless share
collapsing from 0.909 to 0.311. But history is also the only thing that tells the
model whose turn it is and what the conversation is about, and a turn-boundary
judgement made without it is a different task: `exp2_hist0` is *not* a
better-behaved model, it is a model answering an easier question, and its pause
score (0.553) is no better than the full-history baseline's (0.544). Read a
capped-history run as a diagnostic of the warm-up, not as a benchmark result, and
do not compare it against an uncapped one. The prefill caps in `prefill.py`
exist to protect each model's official context window from evicting the
identity frame: PersonaPlex at 2048 frames (NVIDIA trained window), Moshi at
3000 (Kyutai temporal transformer). That is a different job from the
`history_sec` diagnostic knob.

## The pause score is not a measure of turn-taking quality on its own

Across the five conditions above the mean pause score is 0.544 (baseline), 0.526
(drain 15 s), 0.789 (all-PAD text), 0.526 (4 s history) and 0.553 (one frame).
The condition that removes the phonation and the condition that maximises it
score the same, and the **worst** condition scores **best**, because the score
asks only whether the model stayed under the speech gate through the hold and a
model that goes quiet passes. Do not quote a pause score as turn-taking quality
without saying how much of the run was wordless phonation.

`score.json` now carries `speech_ratio` and `wordless_phonation` for every pause
rollout, and `results.json` carries the count of flagged rollouts beside
`mean_score`, using the same `MAX_WORDLESS_SPEECH_RATIO` rule as `bench analyze`
(now defined in `bench/metrics/pause.py`).

The scorer later started treating uttered words as the only evidence of a
response: above-gate murmur with an empty text stream is no longer a takeover.
Quiet still passes. That means the means in this document are **not** comparable
to pause scores produced after that change. Wordless phonation is still counted
beside the mean, not folded into a fail. Gating on text does discard a duplex
model that holds the floor without emitting tokens; that is now an explicit
trade, recorded as `wordless_phonation` rather than as `false_takeover`.

