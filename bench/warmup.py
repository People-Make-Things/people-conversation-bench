"""Spin Modal duplex containers before the first scored rollout.

Holding one websocket per container is enough: each box has max_inputs=1, so
N concurrent connects force N cold starts. Closing them leaves the pool warm
for scaledown_window without changing session state. Hosted models are skipped.

Connections are opened in small waves. A cold herd of 64 handshakes against an
empty pool can sit in the proxy forever; a new connect after the first boxes
are up succeeds in a few hundred milliseconds.
"""

from __future__ import annotations

import asyncio
import resource
import time
from contextlib import asynccontextmanager

import websockets

from bench.protocol import ContextType
from bench.registry import ModelConfig, load_model, resolve_connection

OPEN_FILES_MINIMUM = 4096
WARMUP_OPEN_TIMEOUT_SEC = 90.0
WARMUP_TOTAL_TIMEOUT_SEC = 600.0
WARMUP_STALE_SEC = 90.0
WARMUP_WAVE_SIZE = 8


def raise_open_files(minimum: int = OPEN_FILES_MINIMUM) -> None:
    """Raise the soft NOFILE cap so a wide eval does not run out of sockets."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY:
        target = max(soft, minimum)
    else:
        target = min(max(soft, minimum), hard)
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        print(f"raised RLIMIT_NOFILE {soft} -> {target}", flush=True)
    elif soft < minimum:
        print(
            f"RLIMIT_NOFILE soft={soft} hard={hard}; cannot raise to {minimum}",
            flush=True,
        )


def duplex_socket_url(base: str) -> str:
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    return base.rstrip("/") + "/api/duplex"


@asynccontextmanager
async def wave_connect(url: str, handshake: asyncio.Semaphore):
    """Open one duplex socket; hold the wave semaphore only for the handshake."""
    connect = websockets.connect(
        url,
        max_size=None,
        open_timeout=WARMUP_OPEN_TIMEOUT_SEC,
        close_timeout=1,
        ping_interval=None,
    )
    async with handshake:
        await connect.__aenter__()
    try:
        yield
    finally:
        await connect.__aexit__(None, None, None)


async def warmup_duplex(url: str, count: int, model_id: str) -> int:
    """Hold `count` sockets, connecting a few at a time, then close them all."""
    release = asyncio.Event()
    handshake = asyncio.Semaphore(WARMUP_WAVE_SIZE)
    ready = 0
    lock = asyncio.Lock()
    deadline = time.monotonic() + WARMUP_TOTAL_TIMEOUT_SEC

    async def one() -> bool:
        nonlocal ready
        while time.monotonic() < deadline and not release.is_set():
            try:
                async with wave_connect(url, handshake):
                    async with lock:
                        ready += 1
                        print(f"warming {model_id}: {ready}/{count}", flush=True)
                    await release.wait()
                return True
            except Exception as error:
                print(f"warmup {model_id} connection failed: {error!r}", flush=True)
                await asyncio.sleep(1)
        return False

    started = time.monotonic()
    print(f"warming {model_id}: opening {count} duplex sockets", flush=True)
    tasks = [asyncio.create_task(one()) for _ in range(count)]
    last_ready = -1
    stale_since = time.monotonic()
    while time.monotonic() < deadline:
        async with lock:
            current = ready
        if current >= count:
            break
        if current != last_ready:
            last_ready = current
            stale_since = time.monotonic()
        elif current > 0 and time.monotonic() - stale_since >= WARMUP_STALE_SEC:
            print(
                f"warmup {model_id}: no new containers for {WARMUP_STALE_SEC:.0f}s, "
                f"continuing with {current}/{count}",
                flush=True,
            )
            break
        await asyncio.sleep(0.2)
    release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    warmed = sum(1 for result in results if result is True)
    elapsed = time.monotonic() - started
    print(
        f"warmed {model_id}: {warmed}/{count} containers in {elapsed:.1f}s",
        flush=True,
    )
    return warmed


async def warmup_model(config: ModelConfig, supported: tuple[ContextType, ...]) -> int:
    if ContextType.DUPLEX_AUDIO not in supported:
        return 0
    url = duplex_socket_url(resolve_connection(config))
    return await warmup_duplex(url, config.max_concurrency, config.id)


async def warmup_models(model_refs: list[str]) -> None:
    await asyncio.gather(*(warmup_one(model_ref) for model_ref in model_refs))


async def warmup_one(model_ref: str) -> int:
    config, probe = load_model(model_ref)
    return await warmup_model(config, probe.supported_contexts)
