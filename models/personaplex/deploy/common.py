"""Shared Modal image, volumes, and constants for PersonaPlex deploy apps."""

from __future__ import annotations

from pathlib import Path

import modal

PERSONAPLEX_REPO = "https://github.com/NVIDIA/personaplex.git"
PERSONAPLEX_COMMIT = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
PERSONAPLEX_SUBDIR = "moshi"
HF_SECRET_NAME = "hf-token"
# One session per A10G. Client max_concurrency must match this.
DUPLEX_POOL_SIZE = 64
# Walked rather than indexed with parents[3] because this module is also
# imported from the container, where it sits flat in /root and has no parents[3].
DEPLOY_DIR = Path(__file__).resolve().parent
ROOT = DEPLOY_DIR.parent.parent.parent
SERVER_DIR = DEPLOY_DIR.parent / "server"
BENCH_DIR = ROOT / "bench"
UTILS_DIR = ROOT / "utils"

hf_cache_vol = modal.Volume.from_name("people-bench-hf-cache", create_if_missing=True)
artifacts_vol = modal.Volume.from_name("people-bench-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libopus-dev", "ffmpeg", "git")
    .uv_pip_install(
        "numpy>=1.26,<2.2",
        "safetensors>=0.4.0,<0.5",
        "huggingface-hub>=0.24,<0.25",
        "einops==0.7",
        "sentencepiece==0.2",
        "sphn>=0.1.4,<0.2",
        "torch>=2.2.0,<2.5",
        "aiohttp>=3.10.5,<3.11",
        f"moshi-personaplex @ git+{PERSONAPLEX_REPO}@{PERSONAPLEX_COMMIT}#subdirectory={PERSONAPLEX_SUBDIR}",
    )
    .add_local_dir(str(SERVER_DIR), remote_path="/root/personaplex_server")
    .add_local_dir(str(BENCH_DIR), remote_path="/root/bench")
    .add_local_dir(str(UTILS_DIR), remote_path="/root/utils")
    # Modal only auto-mounts the entrypoint file, so the deploy apps that import
    # this module would otherwise crash-loop on ModuleNotFoundError.
    .add_local_python_source("common")
)

SERVER_PYTHONPATH = "/root/personaplex_server:/root"
