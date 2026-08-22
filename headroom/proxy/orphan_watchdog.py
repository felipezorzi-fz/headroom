"""Orphan watchdog for wrap-spawned proxies.

``headroom wrap`` starts its proxy detached (setsid on POSIX, breakaway
console flags on Windows) on purpose: a crashing wrapper must not tree-kill a
shared proxy out from under other clients. The flip side was a leak: once
every wrapper was gone, nothing ever stopped the proxy, so clientless proxies
accumulated for weeks and later wraps silently reused whatever they found on
the port (port-hijack incident, 2026-08-22). Cleanup on the wrapper side
cannot cover unclean exits, by definition nobody is alive to run it.

When the proxy was spawned by a wrap (``HEADROOM_WRAP_OWNED=1``, set in
``cli/wrap._start_proxy``), this module's background task stops the process
once no live wrap-client markers and no active sessions remain for a grace
period. Standalone ``headroom proxy`` instances and persistent installs never
get the marker env var and are never affected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from headroom._subprocess import identity_mismatch, pid_alive
from headroom.paths import proxy_clients_dir

logger = logging.getLogger(__name__)

WRAP_OWNED_ENV = "HEADROOM_WRAP_OWNED"
GRACE_SECONDS_ENV = "HEADROOM_WRAP_ORPHAN_GRACE_SECONDS"
DEFAULT_GRACE_SECONDS = 900.0
# Floor so a typo ("5" for "5m") cannot make a proxy exit underneath a client
# that simply has not registered its marker yet.
MIN_GRACE_SECONDS = 60.0
DEFAULT_POLL_INTERVAL_SECONDS = 30.0


def orphan_watchdog_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True when this proxy process was spawned by ``headroom wrap``."""
    env = os.environ if env is None else env
    return (env.get(WRAP_OWNED_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def orphan_grace_seconds(env: Mapping[str, str] | None = None) -> float:
    """Grace period with zero live clients before the watchdog exits the proxy."""
    env = os.environ if env is None else env
    raw = env.get(GRACE_SECONDS_ENV)
    if raw:
        try:
            return max(MIN_GRACE_SECONDS, float(raw))
        except ValueError:
            logger.warning(
                "event=orphan_watchdog_bad_grace value=%r default=%.0f", raw, DEFAULT_GRACE_SECONDS
            )
    return DEFAULT_GRACE_SECONDS


def _marker_pid_recycled(marker: Path, pid: int) -> bool:
    """True only if the live ``pid`` is provably a different process than the
    one that wrote ``marker`` (PID recycled after a crash)."""
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    return identity_mismatch(record.get("start_src"), record.get("start_time"), pid)


def live_client_pids(clients_dir: Path) -> list[int]:
    """Live wrap-client PIDs for a clients dir, pruning stale markers as we go.

    Mirrors ``cli.wrap._live_proxy_clients`` without the self-exclusion; the
    proxy is never its own client.
    """
    if not clients_dir.exists():
        return []
    live: list[int] = []
    for marker in clients_dir.glob("*.json"):
        try:
            pid = int(marker.stem)
        except ValueError:
            continue
        if not pid_alive(pid) or _marker_pid_recycled(marker, pid):
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        live.append(pid)
    return live


def _active_session_count(proxy: Any) -> int:
    registry = getattr(proxy, "ws_sessions", None)
    if registry is None:
        return 0
    try:
        return int(registry.active_count())
    except Exception:  # defensive: never let telemetry kill the watchdog
        return 0


def _served_request_count(proxy: Any) -> int | None:
    """Best-effort count of proxied requests served so far, or None.

    Covers clients the marker scheme cannot see: direct HTTP callers and
    clients reaching the loopback listener through an SSH port-forward write
    no wrap-client marker and may hold no WebSocket session. Any proxied LLM
    call allocates a request id, so a changing counter means "in use".
    """
    counter = getattr(proxy, "_request_counter", None)
    if isinstance(counter, int):
        return counter
    return None


def _default_stop() -> None:
    # signal.raise_signal (not os.kill): on Windows CPython maps
    # os.kill(pid, SIGTERM) to TerminateProcess, a hard kill that skips
    # uvicorn's handler and the lifespan teardown; raise_signal dispatches
    # through the emulated handler table, so uvicorn shuts down gracefully on
    # both platforms.
    signal.raise_signal(signal.SIGTERM)


async def orphan_watchdog_loop(
    proxy: Any,
    *,
    grace_seconds: float,
    interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    stop: Callable[[], None] = _default_stop,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Poll wrap-client markers; ``stop()`` once clientless for ``grace_seconds``.

    The grace window covers the boot race (the wrapper registers its marker
    just before spawning the proxy, and re-registers on port fallback) and
    short client restarts. Active WebSocket relay sessions (e.g. a Codex
    session whose wrapper was killed mid-stream) and any change to the
    proxied-request counter (direct HTTP clients, SSH-forwarded clients)
    suppress shutdown even with no live markers. Long idle HTTP streams are
    covered by the grace period: they finish in minutes, the default grace is
    15.
    """
    port = proxy.config.port
    clients_dir = proxy_clients_dir(port)
    idle_since: float | None = None
    last_served = _served_request_count(proxy)
    while True:
        await asyncio.sleep(interval_seconds)
        clients = live_client_pids(clients_dir)
        active = _active_session_count(proxy)
        served = _served_request_count(proxy)
        served_changed = served is not None and last_served is not None and served != last_served
        if served is not None:
            last_served = served
        if clients or active > 0 or served_changed:
            if idle_since is not None:
                logger.info(
                    "event=orphan_watchdog_reset port=%d live_clients=%s active_sessions=%d",
                    port,
                    clients,
                    active,
                )
            idle_since = None
            continue
        now = monotonic()
        if idle_since is None:
            idle_since = now
            continue
        if now - idle_since >= grace_seconds:
            logger.info(
                "event=orphan_watchdog_exit port=%d idle_seconds=%.0f reason=no_live_wrap_clients",
                port,
                now - idle_since,
            )
            stop()
            return
