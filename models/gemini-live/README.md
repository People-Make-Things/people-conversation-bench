# Gemini Live

Google [Gemini 3.1 Flash Live](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-live-preview) over the Live API WebSocket.

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GCP_API_KEY` | Interact / eval | Gemini API key (Google AI Studio) |

## Interact

```bash
uv run bench interact --model gemini-live
```

Override defaults:

```bash
uv run bench interact --model gemini-live --text-prompt "You are concise and helpful."
uv run bench interact --model gemini-live --voice-prompt Puck
uv run bench interact --model gemini-live --preload-point data/processed/example_1/points/000001
```

`--preload-point` loads `context.json` from the eval point (see root README for schema)
and injects completed turns as Live API `clientContent` history.
If the point has `text_prompt`, that becomes session `systemInstruction` unless
`--text-prompt` is passed. Supported context types: `text_audio` (preferred), `text`.

## Eval

Same non-duplex path as GPT Realtime: `rollout_context.json` as text history, then
the live user turn as PCM. EOT points disable server VAD and send `activityEnd`
after the recorded turn.

```bash
uv run bench eval --model gemini-live --point data/processed/example_1/points/000001 --rollouts 1
```
