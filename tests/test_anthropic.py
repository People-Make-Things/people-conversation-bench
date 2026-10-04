"""Tests for the Anthropic JSON helper."""

from __future__ import annotations

import json

from utils.anthropic import anthropic_json_complete, message_text


def test_message_text_ignores_thinking_blocks() -> None:
    text = message_text(
        {
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": '{"correct": false}'},
            ]
        }
    )

    assert text == '{"correct": false}'


def test_anthropic_json_complete_asks_for_a_boolean(monkeypatch) -> None:
    seen: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": '{"correct": true}'}],
                }
            ).encode()

    def urlopen(request, timeout):
        seen["body"] = json.loads(request.data.decode())
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr("utils.anthropic.env.require", lambda _name: "test-key")
    monkeypatch.setattr("utils.anthropic.urllib.request.urlopen", urlopen)

    assert (
        anthropic_json_complete("grade it", "transcript", model="claude-opus-5-5")
        == '{"correct": true}'
    )
    assert "temperature" not in seen["body"]
    assert seen["body"]["model"] == "claude-opus-5-5"
    assert seen["body"]["output_config"]["effort"] == "low"
    schema = seen["body"]["output_config"]["format"]["schema"]
    assert schema["properties"]["correct"]["type"] == "boolean"
    assert schema["required"] == ["correct"]
