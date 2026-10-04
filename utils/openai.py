"""Shared OpenAI chat JSON helper."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import env

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def openai_json_complete(system: str, prompt: str, *, model: str) -> str:
    api_key = env.require("OPENAI_API_KEY")
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        OPENAI_CHAT_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    # Surface API failures as RuntimeError so prepare/eval can skip or record them.
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI JSON request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI JSON request failed: {exc}") from exc
    return str(payload["choices"][0]["message"]["content"])
