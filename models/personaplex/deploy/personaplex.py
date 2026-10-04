"""Deploy PersonaPlex duplex-preload server on Modal with GPU scale-to-zero."""

from __future__ import annotations

import os
import subprocess
import sys

import modal

from common import (
    DUPLEX_POOL_SIZE,
    HF_SECRET_NAME,
    SERVER_PYTHONPATH,
    artifacts_vol,
    hf_cache_vol,
    image,
)

MINUTES = 60
PORT = 8998

app = modal.App(os.environ.get("MODAL_APP_NAME", "people-bench-personaplex"))


@app.function(
    image=image,
    gpu="A100",
    min_containers=0,
    max_containers=DUPLEX_POOL_SIZE,
    scaledown_window=2 * MINUTES,
    timeout=60 * MINUTES,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/artifacts": artifacts_vol,
    },
)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=PORT, startup_timeout=10 * MINUTES)
def serve():
    """Run the custom duplex PersonaPlex server."""
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            f"HF_TOKEN missing. Add it to Modal secret '{HF_SECRET_NAME}'."
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = SERVER_PYTHONPATH
    cmd = [
        sys.executable,
        "/root/personaplex_server/duplex_server.py",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
    ]
    print("starting:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd, env=env)


@app.local_entrypoint()
def main():
    url = serve.get_web_url()
    print(f"PersonaPlex is deployed at {url}")
    print(f"Web UI:        {url}/")
    print(f"Chat socket:   {url.replace('https://', 'wss://')}/api/chat")
    print(f"Duplex socket: {url.replace('https://', 'wss://')}/api/duplex")
