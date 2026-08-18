"""A tiny mock OpenAI-compatible endpoint for offline live-smoke testing.

Run it, point ``OPENAI_API_BASE`` at it, and use a config whose members are ``openai/<anything>``
to exercise the whole running server (uvicorn -> plan -> litellm -> HTTP -> synthesis -> SSE ->
SQLite) without real providers or keys. See ``docs/live-testing.md``.

    python -m tools.mock_openai_server [port]
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys


_MESSAGE = {"role": "assistant", "content": "MOCK reply"}
_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
_NONSTREAM = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "created": 0,
    "model": "mock",
    "choices": [{"index": 0, "message": _MESSAGE, "finish_reason": "stop"}],
    "usage": _USAGE,
}

# Streamed deltas (a terminal finish chunk is appended in the handler).
_STREAM_DELTAS = [
    {"role": "assistant", "content": ""},
    {"content": "MOCK "},
    {"content": "synth"},
]


def _chunk(delta: dict[str, object], finish: str | None) -> bytes:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "mock",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta in _STREAM_DELTAS:
                self.wfile.write(_chunk(delta, None))
                self.wfile.flush()
            self.wfile.write(_chunk({}, "stop"))
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            payload = json.dumps(_NONSTREAM).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9099
    ThreadingHTTPServer(("127.0.0.1", port), _Handler).serve_forever()


if __name__ == "__main__":
    main()
