# Moshi

[Kyutai Moshi](https://github.com/kyutai-labs/moshi) 7B speech-to-speech (Moshiko) on the same duplex WebSocket path as PersonaPlex. The client adapter, Opus codec, and `/api/duplex` server are shared; this folder is the registry entry and the Modal deploy that loads official Kyutai weights.

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | Deploy | Hugging Face token for [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16) |

## Deploy (Modal)

Create a Modal secret named `hf-token` with key `HF_TOKEN`, then:

```bash
uv run modal deploy models/moshi/deploy/serve.py
```

The image is the PersonaPlex image (NVIDIA `moshi-personaplex` loaders, which accept official Moshi weights). The serve command points `--hf-repo` at `kyutai/moshiko-pytorch-bf16` and skips PersonaPlex voice-prompt packs.

Runtime endpoint is fixed in `model.toml`:

`https://peoplemakethings--people-bench-moshi-serve.modal.run`

## Interact

```bash
uv run bench interact --model moshi
```

Preload a processed eval point, then continue live:

```bash
uv run bench interact --model moshi --preload-point data/processed/example_1/points/000001
```

`--preload-point` also loads `point.json` `text_prompt` as the session system prompt unless `--text-prompt` is passed.

Moshi has no `.pt` voice-prompt library. Eval sends no voice prompt; the session voice comes from the teacher-forced assistant history (or the built-in Moshiko voice on a cold live session).

## Eval

Same duplex path as PersonaPlex: preload `rollout_user_context.wav` + `rollout_assistant.wav` (+ transcript), then stream the live user channel. Pause points stream `rollout_user_input_pause_end.wav` plus drain silence; EOT points stream `rollout_user_input.wav` plus silence until the model stops. Do not sleep through those windows — the model steps once per received frame.

History is capped at `MOSHI_MAX_HISTORY_FRAMES` (3000 frames, 240 s) and
truncated from the front. That is Kyutai's temporal-transformer context
([`loaders.py`](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py)
`context=3000`,
[Transformers Moshi docs](https://huggingface.co/docs/transformers/main/model_doc/moshi)
`max_position_embeddings` / `sliding_window` default 3000). PersonaPlex on the
same server uses NVIDIA's 2048-frame trained window instead.

```bash
uv run bench eval --model moshi --point data/processed/example_1/points/000001 --rollouts 1
```
