"""Tests for the wrap-spawned proxy orphan watchdog."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import headroom.proxy.server as server
from headroom.proxy import orphan_watchdog as ow
from headroom.proxy.server import ProxyConfig


def _write_marker(clients_dir: Path, pid: int, **extra: object) -> Path:
    clients_dir.mkdir(parents=True, exist_ok=True)
    marker = clients_dir / f"{pid}.json"
    payload = {"pid": pid, "started_at": 1.0}
    payload.update(extra)
    marker.write_text(json.dumps(payload), encoding="utf-8")
    return marker


class TestEnvConfig:
    def test_enabled_requires_wrap_owned_flag(self, monkeypatch) -> None:
        monkeypatch.delenv(ow.WRAP_OWNED_ENV, raising=False)
        assert ow.orphan_watchdog_enabled() is False
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "1")
        assert ow.orphan_watchdog_enabled() is True
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "0")
        assert ow.orphan_watchdog_enabled() is False
        # Explicit mapping wins over os.environ.
        assert ow.orphan_watchdog_enabled({ow.WRAP_OWNED_ENV: "true"}) is True

    def test_grace_seconds_default_override_and_floor(self, monkeypatch) -> None:
        monkeypatch.delenv(ow.GRACE_SECONDS_ENV, raising=False)
        assert ow.orphan_grace_seconds() == ow.DEFAULT_GRACE_SECONDS
        monkeypatch.setenv(ow.GRACE_SECONDS_ENV, "120")
        assert ow.orphan_grace_seconds() == 120.0
        # Below the floor: clamped, so a typo cannot exit a proxy under a
        # client that has not registered yet.
        monkeypatch.setenv(ow.GRACE_SECONDS_ENV, "5")
        assert ow.orphan_grace_seconds() == ow.MIN_GRACE_SECONDS
        monkeypatch.setenv(ow.GRACE_SECONDS_ENV, "not-a-number")
        assert ow.orphan_grace_seconds() == ow.DEFAULT_GRACE_SECONDS


class TestLiveClientPids:
    def test_live_marker_is_kept(self, tmp_path) -> None:
        _write_marker(tmp_path, os.getpid())
        assert ow.live_client_pids(tmp_path) == [os.getpid()]

    def test_dead_pid_marker_is_pruned(self, tmp_path) -> None:
        # PID 2**31 - 1 is never alive.
        marker = _write_marker(tmp_path, 2**31 - 1)
        assert ow.live_client_pids(tmp_path) == []
        assert not marker.exists()

    def test_recycled_pid_marker_is_pruned(self, tmp_path, monkeypatch) -> None:
        # Live PID (ours) but a recorded identity that provably differs.
        monkeypatch.setattr(ow, "identity_mismatch", lambda src, recorded, pid: True)
        marker = _write_marker(tmp_path, os.getpid(), start_src="psutil", start_time=1.0)
        assert ow.live_client_pids(tmp_path) == []
        assert not marker.exists()

    def test_garbage_and_non_numeric_markers_are_ignored(self, tmp_path) -> None:
        (tmp_path / "not-a-pid.json").write_text("{}", encoding="utf-8")
        (tmp_path / f"{os.getpid()}.json").write_text("not json", encoding="utf-8")
        # Unparseable marker JSON is not proof of recycling: PID is alive, kept.
        assert ow.live_client_pids(tmp_path) == [os.getpid()]

    def test_missing_dir_is_empty(self, tmp_path) -> None:
        assert ow.live_client_pids(tmp_path / "nope") == []


class TestWatchdogLoop:
    @staticmethod
    def _proxy(port: int = 8787, active_sessions: int = 0) -> SimpleNamespace:
        ws = SimpleNamespace(active_count=lambda: active_sessions)
        return SimpleNamespace(config=SimpleNamespace(port=port), ws_sessions=ws)

    def test_stops_after_grace_with_no_clients(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        stopped: list[bool] = []

        asyncio.run(
            ow.orphan_watchdog_loop(
                self._proxy(),
                grace_seconds=0.05,
                interval_seconds=0.01,
                stop=lambda: stopped.append(True),
            )
        )

        assert stopped == [True]

    def test_no_stop_while_client_alive(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        _write_marker(tmp_path, os.getpid())
        stopped: list[bool] = []

        async def run() -> None:
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    self._proxy(),
                    grace_seconds=0.3,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.4)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_no_stop_while_ws_session_active(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        stopped: list[bool] = []

        async def run() -> None:
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    self._proxy(active_sessions=1),
                    grace_seconds=0.3,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.4)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_no_stop_while_requests_are_served(self, tmp_path, monkeypatch) -> None:
        """Direct-HTTP / SSH-forwarded clients leave no marker and may hold no
        WS session; served-request counter movement must still hold the exit."""
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        counter = {"n": 0}
        proxy = SimpleNamespace(
            config=SimpleNamespace(port=8787),
            ws_sessions=None,
            _request_counter=0,
        )
        stopped: list[bool] = []

        async def serve_traffic() -> None:
            while True:
                await asyncio.sleep(0.05)
                counter["n"] += 1
                proxy._request_counter = counter["n"]

        async def run() -> None:
            traffic = asyncio.create_task(serve_traffic())
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    proxy,
                    grace_seconds=0.15,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.4)
            task.cancel()
            traffic.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_grace_restarts_when_client_returns(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        stopped: list[bool] = []

        async def run() -> None:
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    self._proxy(),
                    grace_seconds=0.5,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            # Let the idle clock accumulate, then register a live client: the
            # clock must reset, not fire late.
            await asyncio.sleep(0.2)
            _write_marker(tmp_path, os.getpid())
            await asyncio.sleep(0.7)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_missing_ws_registry_counts_as_idle(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        proxy = SimpleNamespace(config=SimpleNamespace(port=8787), ws_sessions=None)
        stopped: list[bool] = []

        asyncio.run(
            ow.orphan_watchdog_loop(
                proxy,
                grace_seconds=0.05,
                interval_seconds=0.01,
                stop=lambda: stopped.append(True),
            )
        )

        assert stopped == [True]


class TestDefensiveHelpers:
    def test_non_dict_marker_json_is_not_recycling_proof(self, tmp_path) -> None:
        # Valid JSON but not an object: only a dict record can prove recycling.
        marker = _write_marker(tmp_path, os.getpid())
        marker.write_text("[1, 2]", encoding="utf-8")
        assert ow._marker_pid_recycled(marker, os.getpid()) is False

    def test_unlink_oserror_is_tolerated(self, tmp_path, monkeypatch) -> None:
        dead_pid = 2**31 - 1  # never alive
        marker = _write_marker(tmp_path, dead_pid)
        real_unlink = Path.unlink

        def flaky_unlink(self: Path, *args: object, **kwargs: object) -> None:
            if self == marker:
                raise OSError("simulated EPERM")
            real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)
        # The scan tolerates the failed prune and still reports no live clients.
        assert ow.live_client_pids(tmp_path) == []
        assert marker.exists()

    def test_active_session_count_swallows_registry_errors(self) -> None:
        def boom() -> int:
            raise RuntimeError("registry exploded")

        proxy = SimpleNamespace(ws_sessions=SimpleNamespace(active_count=boom))
        assert ow._active_session_count(proxy) == 0

    def test_default_stop_raises_sigterm(self, monkeypatch) -> None:
        sent: list[int] = []
        monkeypatch.setattr(ow.signal, "raise_signal", lambda sig: sent.append(sig))
        ow._default_stop()
        assert sent == [ow.signal.SIGTERM]


class TestServerWiring:
    """create_app must start the watchdog only for wrap-spawned single-worker
    proxies, and cancel it cleanly on shutdown."""

    @staticmethod
    def _config() -> ProxyConfig:
        return ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )

    @staticmethod
    async def _fake_loop(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(3600)

    def test_starts_and_stops_watchdog_when_wrap_owned(self, monkeypatch) -> None:
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "1")
        monkeypatch.delenv(server._MULTI_WORKER_CONFIG_ENV, raising=False)
        monkeypatch.setattr(server, "orphan_watchdog_loop", self._fake_loop)

        app = server.create_app(self._config())
        with TestClient(app) as client:
            task = client.app.state.orphan_watchdog_task
            assert task is not None
            assert not task.done()
        # Lifespan teardown cancelled it and cleared the state slot.
        assert app.state.orphan_watchdog_task is None

    def test_skips_watchdog_for_multi_worker(self, monkeypatch) -> None:
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "1")
        monkeypatch.setenv(server._MULTI_WORKER_CONFIG_ENV, "{}")
        monkeypatch.setattr(server, "orphan_watchdog_loop", self._fake_loop)

        app = server.create_app(self._config())
        with TestClient(app) as client:
            assert client.app.state.orphan_watchdog_task is None

    def test_skips_watchdog_when_not_wrap_owned(self, monkeypatch) -> None:
        monkeypatch.delenv(ow.WRAP_OWNED_ENV, raising=False)
        monkeypatch.delenv(server._MULTI_WORKER_CONFIG_ENV, raising=False)
        monkeypatch.setattr(server, "orphan_watchdog_loop", self._fake_loop)

        app = server.create_app(self._config())
        with TestClient(app) as client:
            assert client.app.state.orphan_watchdog_task is None
