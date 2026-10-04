# GPT Realtime

Shared OpenAI Realtime WebSocket adapter used by two turn-detection profiles:

- `gpt-realtime-fast` — 300 ms VAD silence duration
- `gpt-realtime-slow` — 500 ms VAD silence duration

`grok-voice` uses this same adapter with `[model] session_layout = "xai"` and
`[connection] api_key_env = "XAI_API_KEY"`.

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Interact | OpenAI API key for realtime access |

## Interact

```bash
uv run bench interact --model gpt-realtime-fast
uv run bench interact --model gpt-realtime-slow
```

Override profile defaults:

```bash
uv run bench interact --model gpt-realtime-fast --text-prompt "You are concise and helpful."
uv run bench interact --model gpt-realtime-fast --voice-prompt alloy
uv run bench interact --model gpt-realtime-fast --preload-point data/processed/example_1/points/000001
```

`--preload-point` loads `context.json` from the eval point (see root README for schema),
and injects the conversation history into the Realtime session.
If the point has `text_prompt`, that becomes session `instructions` unless
`--text-prompt` is passed. Supported context types: `text_audio` (preferred), `text`.
