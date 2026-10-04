# Grok Voice

xAI [Grok Voice Agent](https://docs.x.ai/developers/model-capabilities/audio/voice-agent) over the Realtime WebSocket. Uses the shared GPT Realtime adapter (`session_layout = xai`, `api_key_env = XAI_API_KEY`).

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `XAI_API_KEY` | Interact / eval | xAI API key. Worktrees load this from the main checkout `.env`. |

## Interact

```bash
uv run bench interact --model grok-voice
```

Override defaults:

```bash
uv run bench interact --model grok-voice --text-prompt "You are concise and helpful."
uv run bench interact --model grok-voice --voice-prompt ara
uv run bench interact --model grok-voice --preload-point data/processed/example_1/points/000001
```

`--preload-point` injects `context.json` as `conversation.item.create` history,
same as GPT Realtime. If the point has `text_prompt`, that becomes session
`instructions` unless `--text-prompt` is passed.

## Eval

Same non-duplex path as GPT Realtime: `rollout_context.json` as text history, then
the live user turn as PCM.

```bash
uv run bench eval --model grok-voice --point data/processed/example_1/points/000001 --rollouts 1
```
