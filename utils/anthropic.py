"""Shared Anthropic messages JSON helper."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import env

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

CORRECT_SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}},
    "required": ["correct"],
    "additionalProperties": False,
}


def message_text(payload: dict) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        raise RuntimeError("Anthropic JSON response had no content")
    texts = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "".join(texts).strip()
    if not text:
        raise RuntimeError("Anthropic JSON response had no text block")
    return text


def anthropic_json_complete(system: str, prompt: str, *, model: str) -> str:
    api_key = env.require("ANTHROPIC_API_KEY")
    # Opus 5.5 rejects temperature. Low effort is the stable setting for a
    # boolean grade; the default medium spends a thinking pass on it.
    body = {
        "model": model,
        "max_tokens": 4096,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": CORRECT_SCHEMA},
        },
    }
    request = urllib.request.Request(
        ANTHROPIC_MESSAGES_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        method="POST",
    )
    # Surface API failures as RuntimeError so a rescore can retry the file.
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Anthropic JSON request failed ({exc.code}): {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Anthropic JSON request failed: {exc}") from exc
    if payload.get("stop_reason") == "max_tokens":
        raise RuntimeError("Anthropic JSON response was truncated")
    return message_text(payload)
