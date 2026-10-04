"""Deploy Kyutai Moshi on the shared PersonaPlex duplex server."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

PERSONAPLEX_DEPLOY = Path(__file__).resolve().parent.parent.parent / "personaplex" / "deploy"
if str(PERSONAPLEX_DEPLOY) not in sys.path:
    sys.path.insert(0, str(PERSONAPLEX_DEPLOY))

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
MOSHI_HF_REPO = "kyutai/moshiko-pytorch-bf16"

app = modal.App(os.environ.get("MODAL_APP_NAME", "people-bench-moshi"))


def server_command(python: str, port: int) -> list[str]:
    return [
        python,
        "/root/personaplex_server/duplex_server.py",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--hf-repo",
        MOSHI_HF_REPO,
        "--voice-prompt-dir",
        "none",
        "--static",
        "none",
    ]


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
    """Run the shared duplex server with official Kyutai Moshi weights."""
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            f"HF_TOKEN missing. Add it to Modal secret '{HF_SECRET_NAME}'."
        )

    # Modal mounts this model folder onto /root. A file named moshi.py there
    # wins over the installed moshi package because PYTHONPATH includes /root.
    shadow = Path("/root/moshi.py")
    if shadow.exists():
        print(f"removing {shadow} so moshi.models can import", flush=True)
        shadow.unlink()

    env = os.environ.copy()
    env["PYTHONPATH"] = SERVER_PYTHONPATH
    cmd = server_command(sys.executable, PORT)
    print("starting:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd, env=env)


@app.local_entrypoint()
def main():
    url = serve.get_web_url()
    print(f"Moshi is deployed at {url}")
    print(f"Chat socket:   {url.replace('https://', 'wss://')}/api/chat")
    print(f"Duplex socket: {url.replace('https://', 'wss://')}/api/duplex")
